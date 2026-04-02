# Inference with 2 view images
python mhr2smpl/multi_view/infer_two_images.py \
  --image0 multi-view-data/steve_uncalib/1.jpg \
  --image1 multi-view-data/steve_uncalib/2.jpg \
  --focal0 1200 --focal1 1200 \
  --use_smoother \
  --output_dir multi-view-output/steve_uncalib