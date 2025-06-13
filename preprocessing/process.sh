TARGET_PATH="../data/imagenet"

torchrun --nproc_per_node=8 preprocess.py \
    --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=${TARGET_PATH}/vae-sd \
    --dest-images=${TARGET_PATH}/images \
    --batch-size=128 \
    --resolution=256 \
    --transform=center-crop-dhariwal