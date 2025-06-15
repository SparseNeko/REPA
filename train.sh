accelerate launch train.py \
  --report-to="tensorboard" \
  --allow-tf32 \
  --mixed-precision="bf16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=8 \
  --output-dir="exps" \
  --exp-name="linear-dinov2-b-enc8-in512-sta242488-dt" \
  --resolution=512 \
  --num-hidden-layers=24 \
  --hidden-size=1024 \
  --num-heads=16 \
  --attn-num-heads=16 \
  --patch-size=2 \
  --attn-type="sta2d_attn" \
  --batch-size=128 \
  --learning-rate=1.5e-3 \
  --attn_layers="0,2,4,6,8,10,12,14,16,18,20,22" \
  --learning-rate=2e-3 \
  --data-dir="./data/imagenet"

python nvi.py