# Inference with 2 view images
python mhr2smpl/multi_view/infer_two_images.py \
  --image0 multi-view-data/steve/front.jpg \
  --image1 multi-view-data/steve/left.jpg \
  --focal0 1200 --focal1 1200 \
  --output_dir multi-view-data/steve