import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist

from cn_clip.clip.model import convert_state_dict


def is_master(args):
    return args.rank == 0


def get_loss(
    model, images, texts, loss_img, loss_txt, args, accum_image_features=None, accum_text_features=None, accum_idx=-1
):
    """
    计算损失
    """
    # 根据梯度累计频率，计算图像和文本特征
    if args.accum_freq == 1:
        image_features, text_features, logit_scale = model(images, texts, args.mask_ratio)
    else:
        assert accum_image_features and accum_text_features and accum_idx != -1
        chunk_image_features, chunk_text_features, logit_scale = model(images, texts, args.mask_ratio)
        # 叠加以前的特征
        image_features = torch.cat(
            accum_image_features[:accum_idx] + [chunk_image_features] + accum_image_features[accum_idx + 1 :]
        )
        text_features = torch.cat(
            accum_text_features[:accum_idx] + [chunk_text_features] + accum_text_features[accum_idx + 1 :]
        )
    logit_scale = logit_scale.mean()
    if args.aggregate:
        # 在不同的GPU上聚合特征
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        if args.gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)

            all_image_features = torch.cat(
                [image_features] + gathered_image_features[:rank] + gathered_image_features[rank + 1 :]
            )
            all_text_features = torch.cat(
                [text_features] + gathered_text_features[:rank] + gathered_text_features[rank + 1 :]
            )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

    else:
        # 每一行是一个图片, 每一列是一个文本和图片的相似度. shape: (image_batch_size, text_batch_size)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    # 真实标签, shape: (image_batch_size)
    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)

    # 计算图片的损失和文本的损失, 然后取平均
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

    acc = None
    if args.report_training_batch_acc:
        # 计算准确率, 就是在每一行中选择最大的值, 然后看看这个值的索引是不是和真实标签一样
        i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}

    return total_loss, acc


def freeze_vision_bn(args, model):
    """
    冻结视觉部分的参数
    """
    # freeze bn running mean and variance
    if "RN" in args.vision_model:
        RN_visual_modules = (
            model.module.visual.modules()
            if isinstance(model, nn.parallel.DistributedDataParallel)
            else model.visual.modules()
        )
        for m in RN_visual_modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps):
    """
    训练流程
    """
    # os.environ["WDS_EPOCH"] = str(epoch)

    model.train()
    if args.freeze_vision:
        freeze_vision_bn(args, model)

    dataloader, sampler = data["train"].dataloader, data["train"].sampler

    # 图片和文本的损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # 损失函数也要放到 GPU 上
    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    if sampler is not None:
        sampler.set_epoch(epoch)

    # 每个 epoch 的训练步数
    num_steps_per_epoch = dataloader.num_batches // args.accum_freq
    data_iter = iter(dataloader)

    # 如果使用了梯度累积, 那么需要保存一些中间结果
    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

    end = time.time()
    epoch_trained_steps = 0
    # 开始训练每一个 batch
    for i in range(0, dataloader.num_batches):
        batch = next(data_iter)

        i_accum = i // args.accum_freq
        # 当前的步数
        step = num_steps_per_epoch * epoch + i_accum
        # reach the args.max_steps, exit training:
        if step >= args.max_steps:
            logging.info(
                "Stopping training due to step {} has reached max_steps {}".format(
                    step, args.max_steps // args.accum_freq
                )
            )
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        # 获取训练数据
        # images 的 shape 是 (batch_size, 3, 224, 224)
        # texts 的 shape 是 (batch_size, seq_len)
        # eos_indices 的 shape 是 (batch_size, )
        images, texts, eos_indices = batch

        images = images.cuda(args.local_device_rank, non_blocking=True)
        texts = texts.cuda(args.local_device_rank, non_blocking=True)
        eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

        # 计算数据加载时间
        data_time = time.time() - end

        # 实际的模型
        m = model.module

        if args.accum_freq == 1:
            # with automatic mixed precision. 使用混合精度训练
            if args.precision == "amp":
                with autocast():
                    total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                scaler.update()

            else:
                # 不使用混合精度训练
                total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
                total_loss.backward()
                optimizer.step()
        else:
            # 使用梯度累加, 先缓存特征, 然后在第 N 次中计算梯度
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast(enabled=(args.precision == "amp")):
                    chunk_image_features, chunk_text_features, _ = model(images, texts)
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                # 把原始数据也保存起来
                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # 特征累积完了, 可以计算一次了
            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                # 当前的图片和文本输入
                images = accum_images[j]
                texts = accum_texts[j]
                # 计算损失并反向传播
                with autocast(enabled=(args.precision == "amp")):
                    # `total_loss` and `acc` are coarsely sampled, taking only the last result in the loop.
                    # Although each result should be the same in theory, it will be slightly different in practice
                    total_loss, acc = get_loss(
                        model, images, texts, loss_img, loss_txt, args, accum_image_features, accum_text_features, j
                    )
                if args.precision == "amp":
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

            # 等所有的梯度都反向传播, 再更新优化器
            if args.precision == "amp":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # 重置数据
        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []

        # 限制 logit_scale 的范围
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        # 批次时间是总的, 包含前面的数据加载时间
        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        # 打印日志
        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            batch_size = len(images) * args.accum_freq
            num_samples = (i_accum + 1) * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i_accum + 1) / num_steps_per_epoch

            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | "
                + f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | "
                + f"Loss: {total_loss.item():.6f} | "
                + (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "")
                + (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "")
                + f"Data Time: {data_time:.3f}s | "
                + f"Batch Time: {batch_time:.3f}s | "
                + f"LR: {optimizer.param_groups[0]['lr']:5f} | "
                + f"logit_scale: {m.logit_scale.data:.3f} | "
                + f"Global Batch Size: {batch_size * args.world_size}"
            )

        # 奇怪, train 里面也有评估和保存检查点, 外面也有. 哦, 条件不一样, 这里是 valid_step_interval
        # 训练中的评估
        if (
            args.val_data is not None
            and args.valid_step_interval is not None
            and ((step + 1) % args.valid_step_interval) == 0
        ):
            assert "val" in data, "Error: Valid dataset has not been built."
            if not args.use_flash_attention:
                evaluate(model, data, epoch, args, step + 1)
            else:
                # fp16 is needed in flash attention
                with autocast():
                    evaluate(model, data, epoch, args, step + 1)
            # set model back to train mode
            model.train()
            if args.freeze_vision:
                freeze_vision_bn(args, model)

        # 保存模型. 这里的条件是 save_step_frequency, 那个脚本里写了很大的数值, 一般就不会触发了
        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict()
                    if not args.use_flash_attention
                    else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(
                    save_path, epoch + 1, step + 1, time.time() - t1
                )
            )

            # 再保存一遍, 但是检查点名字是固定的
            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict()
                    if not args.use_flash_attention
                    else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(
                    save_path, epoch + 1, step + 1, time.time() - t1
                )
            )

    return epoch_trained_steps


def evaluate(model, data, epoch, args, steps):
    """
    TODO: 看下评估流程
    """
    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))

    model.eval()

    dataloader = data["val"].dataloader
    data_iter = iter(dataloader)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    cumulative_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_i2t_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_t2i_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    num_elements = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for i in range(dataloader.num_batches):
            batch = next(data_iter)
            images, texts, eos_indices = batch

            images = images.cuda(args.local_device_rank, non_blocking=True)
            texts = texts.cuda(args.local_device_rank, non_blocking=True)
            eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

            cumulative_i2t_acc += ((logits_per_image.argmax(-1) == ground_truth).sum()).float()
            cumulative_t2i_acc += (logits_per_text.argmax(-1) == ground_truth).sum().float()

            if (i + 1) % 100 == 0:
                logging.info("Evaluated {}/{} batches...".format(i + 1, dataloader.num_batches))

        dist.all_reduce(cumulative_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_i2t_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_t2i_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_elements, op=dist.ReduceOp.SUM)
        loss = cumulative_loss / num_elements
        i2t_acc = cumulative_i2t_acc / num_elements
        t2i_acc = cumulative_t2i_acc / num_elements

        assert num_elements.item() == dataloader.num_samples  # sanity check

        logging.info(
            f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
            f"Valid Loss: {loss.item():.6f} | "
            f"Image2Text Acc: {i2t_acc.item() * 100:.2f} | "
            f"Text2Image Acc: {t2i_acc.item() * 100:.2f} | "
            f"logit_scale: {model.module.logit_scale.data:.3f} | "
            f"Valid Batch Size: {batch_size}"
        )
