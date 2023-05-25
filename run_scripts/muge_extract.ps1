#  提取特征
$env:CUDA_VISIBLE_DEVICES=0
$env:PYTHONPATH = $env:PYTHONPATH + ";$pwd/cn_clip/"

$DATAPATH=$args[0]
Write-Host $DATAPATH

$split="test" # 指定计算valid或test集特征
$resume="${DATAPATH}/experiments/muge_finetune_vit-b-16_roberta-base_bs160/checkpoints/epoch_latest.pt"

# 中文无法输出
& Write-Host "=============extract_features"
# 提取特征
& Write-Host python -u cn_clip/eval/extract_features.py `
    --extract-image-feats `
    --extract-text-feats `
    --image-data="${DATAPATH}/datasets/MUGE/lmdb/${split}/imgs" `
    --text-data="${DATAPATH}/datasets/MUGE/${split}_texts.jsonl" `
    --img-batch-size=256 `
    --text-batch-size=256 `
    --context-length=52 `
    --resume=${resume} `
    --vision-model=ViT-B-16 `
    --text-model=RoBERTa-wwm-ext-base-chinese

& Write-Host "=============make_topk_predictions"
# 获取 topk 结果. 32768 不知道怎么来的
& Write-Host python -u cn_clip/eval/make_topk_predictions.py `
   --image-feats="${DATAPATH}/datasets/MUGE/${split}_imgs.img_feat.jsonl" `
   --text-feats="${DATAPATH}/datasets/MUGE/${split}_texts.txt_feat.jsonl" `
   --top-k=10 `
   --eval-batch-size=32768 `
   --output="${DATAPATH}/datasets/MUGE/${split}_predictions.jsonl"

& Write-Host "=============evaluation"
# 评估
& Write-Host python cn_clip/eval/evaluation.py `
   ${DATAPATH}/datasets/MUGE/${split}_texts.jsonl `
   ${DATAPATH}/datasets/MUGE/${split}_predictions.jsonl `
   recall_output_${split}.json

# 运行应该是 .\run_scripts\muge_extract.ps1 .\datapath\
