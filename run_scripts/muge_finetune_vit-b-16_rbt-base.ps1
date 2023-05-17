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
$name="muge_finetune_vit-b-16_roberta-base_bs128_1gpu"
$save_step_frequency=999999 # disable it
$save_epoch_frequency=1
$log_interval=1
$report_training_batch_acc="--report-training-batch-acc" #
# report_training_batch_acc=""

# training hyper-params
$context_length=52
$warmup=100
$batch_size=128
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
          $text_model

# 输出是 python -m torch.distributed.launch --use_env --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=8514 cn_clip/training/main.py --train-data=./datapath/datasets/MUGE/lmdb/train --val-data=./datapath/datasets/MUGE/lmdb/valid --resume=./datapath/pretrained_weights/clip_cn_vit-b-16.pt --reset-data-offset --reset-optimizer --logs=./datapath/experiments/ --name=muge_finetune_vit-b-16_roberta-base_bs128_1gpu --save-step-frequency=999999 --save-epoch-frequency=1 --log-interval=1 --report-training-batch-acc --context-length=52 --warmup=100 --batch-size=128 --valid-batch-size=128 --valid-step-interval=150 --valid-epoch-interval=1 --accum-freq=1 --lr=5E-05 --wd=0.001 --max-epochs=3 --vision-model=ViT-B-16 --use-augment --text-model=RoBERTa-wwm-ext-base-chinese

# 用了 --use-flash-attention 没降低显存. 每批次时间从 1.2 左右降低到了 1.0 左右. 在 batch_size=160 时
# python -m torch.distributed.launch --use_env --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=8514 cn_clip/training/main.py --train-data=./datapath/datasets/MUGE/lmdb/train --val-data=./datapath/datasets/MUGE/lmdb/valid --resume=./datapath/pretrained_weights/clip_cn_vit-b-16.pt --reset-data-offset --reset-optimizer --logs=./datapath/experiments/ --name=muge_finetune_vit-b-16_roberta-base_bs128_1gpu --save-step-frequency=999999 --save-epoch-frequency=1 --log-interval=1 --report-training-batch-acc --context-length=52 --warmup=100 --batch-size=160 --valid-batch-size=160 --valid-step-interval=150 --valid-epoch-interval=1 --accum-freq=1 --lr=3E-06 --wd=0.001 --max-epochs=3 --vision-model=ViT-B-16 --use-augment --text-model=RoBERTa-wwm-ext-base-chinese --use-flash-attention
