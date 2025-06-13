TARGET_PATH="../data/imagenet"
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
python dataset_tools.py convert --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=${TARGET_PATH}/images --resolution=512x512 --transform=center-crop-dhariwal

# Convert the pixel data to VAE latents
python dataset_tools.py encode --source=${TARGET_PATH}/images \
    --dest=${TARGET_PATH}/vae-sd