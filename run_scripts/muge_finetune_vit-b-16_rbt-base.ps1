#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# Command: bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}

# Number of GPUs per GPU worker
$GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
$WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
$MASTER_ADDR="localhost"
# The port for communication
$MASTER_PORT=8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
$RANK=0 
$env:PYTHONPATH = $env:PYTHONPATH + ";$pwd/cn_clip/"

$DATAPATH=$args[0]

Write-Host $DATAPATH

# data options
$train_data = "$DATAPATH/datasets/MUGE/lmdb/train"
$val_data = "$DATAPATH/datasets/MUGE/lmdb/valid" # if val_data is not specified, the validation will be automatically disabled

# restore options
$resume = "$DATAPATH/pretrained_weights/clip_cn_vit-b-16.pt"
$reset_data_offset = "--reset-data-offset"
$reset_optimizer = "--reset-optimizer"
# reset_optimizer=""

# output options
$output_base_dir="$DATAPATH/experiments/"
$name="muge_finetune_vit-b-16_roberta-base_bs600_1gpu"
$save_step_frequency=999999 # disable it
$save_epoch_frequency=1
$log_interval=1
$report_training_batch_acc="--report-training-batch-acc" #
# report_training_batch_acc=""

# training hyper-params
$context_length=52
$warmup=100
$batch_size=800
$valid_batch_size=128
$accum_freq=1
$lr=5e-5
$wd=0.001
$max_epochs=3 # or you can alternatively specify --max-steps
$valid_step_interval=150
$valid_epoch_interval=1
$vision_model="ViT-B-16"
$text_model="RoBERTa-wwm-ext-base-chinese"
$use_augment="--use-augment"
# use_augment=""
$grad_checkpointing = "--grad-checkpointing"
$use_flash_attention = "--use-flash-attention"

$python3 = "python"
$torch_distributed_launch = "-m torch.distributed.launch"
$use_env = "--use_env"
$nproc_per_node = "--nproc_per_node=${GPUS_PER_NODE}"
$nnodes = "--nnodes=${WORKER_CNT}"
$node_rank = "--node_rank=${RANK}"
$master_addr = "--master_addr=${MASTER_ADDR}"
$master_port = "--master_port=${MASTER_PORT}"
$cn_clip_training_main = "cn_clip/training/main.py"
$train_data = "--train-data=${train_data}"
$val_data = "--val-data=${val_data}"
$resume = "--resume=${resume}"
$reset_data_offset = "${reset_data_offset}"
$reset_optimizer = "${reset_optimizer}"
$logs = "--logs=${output_base_dir}"
$name = "--name=${name}"
$save_step_frequency = "--save-step-frequency=${save_step_frequency}"
$save_epoch_frequency = "--save-epoch-frequency=${save_epoch_frequency}"
$log_interval = "--log-interval=${log_interval}"
$report_training_batch_acc = "${report_training_batch_acc}"
$context_length = "--context-length=${context_length}"
$warmup = "--warmup=${warmup}"
$batch_size = "--batch-size=${batch_size}"
$valid_batch_size = "--valid-batch-size=${valid_batch_size}"
$valid_step_interval = "--valid-step-interval=${valid_step_interval}"
$valid_epoch_interval = "--valid-epoch-interval=${valid_epoch_interval}"
$accum_freq = "--accum-freq=${accum_freq}"
$lr = "--lr=${lr}"
$wd = "--wd=${wd}"
$max_epochs = "--max-epochs=${max_epochs}"
$vision_model = "--vision-model=${vision_model}"
$use_augment = "${use_augment}"
$grad_checkpointing = "${grad_checkpointing}"
$use_flash_attention = "${use_flash_attention}"
$text_model = "--text-model=${text_model}"
& Write-Host $python3 $torch_distributed_launch $use_env $nproc_per_node $nnodes $node_rank `
        $master_addr $master_port $cn_clip_training_main `
        $train_data `
        $val_data `
        $resume `
        $reset_data_offset `
        $reset_optimizer `
        $logs `
        $name `
        $save_step_frequency `
        $save_epoch_frequency `
        $log_interval `
        $report_training_batch_acc `
        $context_length `
        $warmup `
        $batch_size `
        $valid_batch_size `
        $valid_step_interval `
        $valid_epoch_interval `
        $accum_freq `
        $lr `
        $wd `
        $max_epochs `
        $vision_model `
        $use_augment `
        $text_model `
        $grad_checkpointing `
        $use_flash_attention `

# 运行应该是 .\run_scripts\muge_finetune_vit-b-16_rbt-base.ps1 .\datapath\

# 输出是 python -m torch.distributed.launch --use_env --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=8514 cn_clip/training/main.py --train-data=./datapath/datasets/MUGE/lmdb/train --val-data=./datapath/datasets/MUGE/lmdb/valid --resume=./datapath/pretrained_weights/clip_cn_vit-b-16.pt --reset-data-offset --reset-optimizer --logs=./datapath/experiments/ --name=muge_finetune_vit-b-16_roberta-base_bs128_1gpu --save-step-frequency=999999 --save-epoch-frequency=1 --log-interval=1 --report-training-batch-acc --context-length=52 --warmup=100 --batch-size=128 --valid-batch-size=128 --valid-step-interval=150 --valid-epoch-interval=1 --accum-freq=1 --lr=5E-05 --wd=0.001 --max-epochs=3 --vision-model=ViT-B-16 --use-augment --text-model=RoBERTa-wwm-ext-base-chinese

# 用了 --use-flash-attention 没降低显存. 每批次时间从 1.2 左右降低到了 1.0 左右. 在 batch_size=160 时
# python -m torch.distributed.launch --use_env --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=8514 cn_clip/training/main.py --train-data=./datapath/datasets/MUGE/lmdb/train --val-data=./datapath/datasets/MUGE/lmdb/valid --resume=./datapath/pretrained_weights/clip_cn_vit-b-16.pt --reset-data-offset --reset-optimizer --logs=./datapath/experiments/ --name=muge_finetune_vit-b-16_roberta-base_bs128_1gpu --save-step-frequency=999999 --save-epoch-frequency=1 --log-interval=1 --report-training-batch-acc --context-length=52 --warmup=100 --batch-size=160 --valid-batch-size=160 --valid-step-interval=150 --valid-epoch-interval=1 --accum-freq=1 --lr=3E-06 --wd=0.001 --max-epochs=3 --vision-model=ViT-B-16 --use-augment --text-model=RoBERTa-wwm-ext-base-chinese --use-flash-attention

# 我在 autodl 上租了个 RTX A4000, 才 16GB 显存, 能跑到 400 的 batch_size. 但是在我自己的 RTX 3090 上, 24GB 显存, 只能跑到 160 的 batch_size.
# 原来是使用了 grad_checkpointing, 能极大减少显存, 跑到 600 的 batch_size 才用了 18 GB 左右的显存.

<#
RTX A4000, 16GB 显存, batch_size=400
2023-05-28,11:44:30 | INFO | Rank 0 | Global Steps: 90/626 | Train Epoch: 1 [36000/250400 (14%)] | Loss: 1.363617 | Image2Text Acc: 65.25 | Text2Image Acc: 64.50 | Data Time: 0.071s | Batch Time: 3.672s | LR: 0.000003 | logit_scale: 4.605 | Global Batch Size: 400
2023-05-28,11:45:07 | INFO | Rank 0 | Global Steps: 100/626 | Train Epoch: 1 [40000/250400 (16%)] | Loss: 1.449286 | Image2Text Acc: 62.50 | Text2Image Acc: 62.00 | Data Time: 0.079s | Batch Time: 3.680s | LR: 0.000003 | logit_scale: 4.605 | Global Batch Size: 400

本地 windows, RTX 3090, 24GB 显存, batch_size=160
2023-05-17,21:15:08 | INFO | Rank 0 | Global Steps: 6/4695 | Train Epoch: 1 [960/250400 (0%)] | Loss: 1.137266 | Image2Text Acc: 75.63 | Text2Image Acc: 76.25 | Data Time: 0.358s | Batch Time: 0.959s | LR: 0.000000 | logit_scale: 4.605 | Global Batch Size: 160
2023-05-17,21:15:09 | INFO | Rank 0 | Global Steps: 7/4695 | Train Epoch: 1 [1120/250400 (0%)] | Loss: 1.278120 | Image2Text Acc: 69.38 | Text2Image Acc: 68.13 | Data Time: 0.381s | Batch Time: 0.984s | LR: 0.000000 | logit_scale: 4.605 | Global Batch Size: 160

RTX 3090, 24GB 显存, batch_size=400
2023-05-28,12:03:45 | INFO | Rank 0 | Global Steps: 80/626 | Train Epoch: 1 [32000/250400 (13%)] | Loss: 5.995152 | Image2Text Acc: 0.75 | Text2Image Acc: 0.00 | Data Time: 0.100s | Batch Time: 2.272s | LR: 0.000002 | logit_scale: 2.659 | Global Batch Size: 400
2023-05-28,12:04:08 | INFO | Rank 0 | Global Steps: 90/626 | Train Epoch: 1 [36000/250400 (14%)] | Loss: 6.003726 | Image2Text Acc: 0.75 | Text2Image Acc: 0.25 | Data Time: 0.096s | Batch Time: 2.270s | LR: 0.000003 | logit_scale: 2.659 | Global Batch Size: 400

本地 windows, RTX 3090, 24GB 显存, batch_size=400. 但比较起来也很慢, Data Time 和 Batch Time 都比 autodl 上的 3090 慢. 看监控, 显卡使用率忽高忽低. 主要是 Data Time 比较慢, 导致总的 Batch Time 更高.
2023-05-28,13:10:02 | INFO | Rank 0 | Global Steps: 15/1878 | Train Epoch: 1 [6000/250400 (2%)] | Loss: 1.570396 | Image2Text Acc: 61.75 | Text2Image Acc: 60.25 | Data Time: 1.195s | Batch Time: 3.039s | LR: 0.000008 | logit_scale: 4.605 | Global Batch Size: 400
2023-05-28,13:10:05 | INFO | Rank 0 | Global Steps: 16/1878 | Train Epoch: 1 [6400/250400 (3%)] | Loss: 1.364061 | Image2Text Acc: 65.00 | Text2Image Acc: 65.75 | Data Time: 1.120s | Batch Time: 2.965s | LR: 0.000008 | logit_scale: 4.605 | Global Batch Size: 400

本地 windows, RTX 3090, 24GB 显存, batch_size=800. 最高能到 800, 用了 22.4GB 显存.
2023-05-28,16:26:19 | INFO | Rank 0 | Global Steps: 11/939 | Train Epoch: 1 [8800/250400 (4%)] | Loss: 1.885316 | Image2Text Acc: 53.75 | Text2Image Acc: 54.00 | Data Time: 0.228s | Batch Time: 3.968s | LR: 0.000006 | logit_scale: 4.605 | Global Batch Size: 800
2023-05-28,16:26:23 | INFO | Rank 0 | Global Steps: 12/939 | Train Epoch: 1 [9600/250400 (4%)] | Loss: 1.881319 | Image2Text Acc: 55.25 | Text2Image Acc: 54.12 | Data Time: 0.226s | Batch Time: 4.012s | LR: 0.000006 | logit_scale: 4.605 | Global Batch Size: 800

A40, 48GB 显存, batch_size=2400, 用了 46.9 GB 显存. 这也太夸张了, 只多了 24 GB 显存, 能多这么多的 batch_size
2023-05-28,17:51:54 | INFO | Rank 0 | Global Steps: 12/105 | Train Epoch: 1 [28800/252000 (11%)] | Loss: 2.976864 | Image2Text Acc: 38.33 | Text2Image Acc: 38.63 | Data Time: 0.504s | Batch Time: 10.613s | LR: 0.000000 | logit_scale: 4.605 | Global Batch Size: 2400
2023-05-28,17:52:05 | INFO | Rank 0 | Global Steps: 13/105 | Train Epoch: 1 [31200/252000 (12%)] | Loss: 3.040416 | Image2Text Acc: 37.13 | Text2Image Acc: 38.83 | Data Time: 0.558s | Batch Time: 10.657s | LR: 0.000000 | logit_scale: 4.605 | Global Batch Size: 2400

#>

