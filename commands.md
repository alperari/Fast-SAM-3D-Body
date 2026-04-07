# Custom inference with multi view images
python mhr2smpl/multi_view/infer_multiview_custom.py \
  --input_dir multi-view-data/steve_uncalib_4v \
  --uncalibrated \
  --use_smoother \
  --output_dir multi-view-output/steve_uncalib_4v


# Custom inference saving MHR mesh of each view
python mhr2smpl/multi_view/infer_multiview_custom.py \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --input_dir multi-view-data/mpi_1v \
  --uncalibrated \
  --save_mhr_mesh \
  --output_dir multi-view-output/mpi_1v
