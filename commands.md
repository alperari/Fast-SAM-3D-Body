# Inference with 2 view images
python mhr2smpl/multi_view/infer_multiview_custom.py \
  --input_dir multi-view-data/steve_uncalib_4v \
  --uncalibrated \
  --use_smoother \
  --output_dir multi-view-output/steve_uncalib_4v