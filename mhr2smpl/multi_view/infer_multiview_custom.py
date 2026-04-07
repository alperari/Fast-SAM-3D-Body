#!/usr/bin/env python3
"""
Run multi-view MHR2SMPL inference from N images in a folder.

Pipeline:
  image_dir (N camera views of the same frame)
    -> SAM-3D-Body Stage1 per image (pred_vertices, pred_cam_t)
    -> MHR2SMPL multi-view fusion
    -> Mesh export (.obj): SMPL and/or SAM-3D-Body MHR + parameters (.npz)

Example:
  python mhr2smpl/multi_view/infer_two_images.py \
      --input_dir /path/to/multiview_images \
      --intrinsics_dir /path/to/intrinsics \
      --output_dir mhr2smpl/output_multi_images

Uncalibrated quick-start:
  python mhr2smpl/multi_view/infer_two_images.py \
      --input_dir /path/to/multiview_images \
      --uncalibrated \
      --output_dir mhr2smpl/output_multi_images_uncalib
"""
# fmt: off
from infer_multiview import MHR2SMPLMultiView
from pathlib import Path
import sys

# Ensure project root and script dir are on sys.path before importing local packages
SCRIPT_DIR = Path(__file__).parent.resolve()
MHR2SMPL_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MHR2SMPL_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
import argparse
import inspect
import json
import os

import cv2
import numpy as np
import torch
# fmt: on


def parse_bbox(text: str | None) -> np.ndarray | None:
    if text is None:
        return None
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 4:
        raise ValueError(
            f"bbox must have 4 comma-separated values, got: {text}")
    return np.asarray(vals, dtype=np.float32).reshape(1, 4)


def build_default_cam_int(width: int, height: int, focal: float) -> np.ndarray:
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def estimate_focal_from_fov(width: int, height: int, fov_deg: float) -> float:
    # Approximate pinhole focal from horizontal FOV and image width.
    if fov_deg <= 1.0 or fov_deg >= 179.0:
        raise ValueError(f"default_fov_deg must be in (1, 179), got {fov_deg}")
    half_fov_rad = np.deg2rad(fov_deg * 0.5)
    return float((width * 0.5) / np.tan(half_fov_rad))


def list_images(input_dir: str, pattern: str) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input_dir is not a directory: {root}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        [p for p in root.glob(pattern) if p.is_file()
         and p.suffix.lower() in exts]
    )
    if len(images) == 0:
        raise FileNotFoundError(
            f"no images found in {root} with pattern='{pattern}' and supported extensions {sorted(exts)}"
        )
    return images


def find_intrinsics_for_image(image_path: Path, intrinsics_dir: str | None) -> str | None:
    if intrinsics_dir is None:
        return None
    intr_dir = Path(intrinsics_dir)
    if not intr_dir.is_dir():
        raise FileNotFoundError(
            f"intrinsics_dir is not a directory: {intr_dir}")

    for ext in (".json", ".npy", ".npz"):
        cand = intr_dir / f"{image_path.stem}{ext}"
        if cand.exists():
            return str(cand)
    return None


def load_cam_int(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"intrinsics file not found: {p}")

    if p.suffix.lower() == ".npy":
        cam_int = np.load(p).astype(np.float32)
    elif p.suffix.lower() == ".npz":
        d = np.load(p)
        key = "camera_matrix" if "camera_matrix" in d else "cam_int"
        if key not in d:
            raise KeyError(f"{p} must contain 'camera_matrix' or 'cam_int'")
        cam_int = np.asarray(d[key], dtype=np.float32)
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        key = "camera_matrix" if "camera_matrix" in d else "cam_int"
        if key not in d:
            raise KeyError(f"{p} must contain 'camera_matrix' or 'cam_int'")
        cam_int = np.asarray(d[key], dtype=np.float32)
    else:
        raise ValueError(f"unsupported intrinsics format: {p.suffix}")

    if cam_int.shape == (1, 3, 3):
        cam_int = cam_int[0]
    if cam_int.shape != (3, 3):
        raise ValueError(
            f"intrinsics must be [3,3] or [1,3,3], got {cam_int.shape}")
    return cam_int


def pick_main_prediction(preds: list[dict]) -> dict:
    if len(preds) == 0:
        raise RuntimeError("no person detected")
    if len(preds) == 1:
        return preds[0]
    areas = []
    for p in preds:
        bbox = np.asarray(p.get("bbox", [0, 0, 1, 1]), dtype=np.float32)
        areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return preds[int(np.argmax(np.asarray(areas)))]


def save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # OBJ is 1-indexed
        for tri in faces:
            f.write(
                f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def infer_single_view(
    estimator,
    image_path: str,
    bbox: np.ndarray | None,
    cam_int: np.ndarray,
    hand_box_source: str,
):
    cam_int_t = torch.from_numpy(cam_int.astype(np.float32)).unsqueeze(0)
    preds = estimator.process_one_image(
        image_path,
        bboxes=bbox,
        cam_int=cam_int_t,
        hand_box_source=hand_box_source,
    )
    return pick_main_prediction(preds)


def resolve_detector_model_path(detector: str, detector_model: str | None) -> str | None:
    if detector not in ("yolo_pose", "yolo", 'vitdet'):
        return detector_model
    if detector_model and Path(detector_model).exists():
        return detector_model

    yolo_dir = PROJECT_DIR / "checkpoints/yolo"
    if detector == "yolo_pose":
        candidates = [
            yolo_dir / "yolo11m-pose.engine",
            yolo_dir / "yolo11m-pose.pt",
            yolo_dir / "yolo11n-pose.pt",
        ]
    else:
        candidates = [
            yolo_dir / "yolo11m.engine",
            yolo_dir / "yolo11m.pt",
            yolo_dir / "yolo11n.pt",
        ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def patch_chumpy_compat() -> None:
    """Patch Python/NumPy compatibility for chumpy used by smplx .pkl loading."""
    if not hasattr(inspect, "getargspec"):
        # type: ignore[attr-defined]
        inspect.getargspec = inspect.getfullargspec
    for alias, target in (
        ("bool", np.bool_),
        ("int", np.int_),
        ("float", np.float64),
        ("complex", np.complex128),
        ("object", np.object_),
        ("str", np.str_),
    ):
        if alias not in np.__dict__:
            setattr(np, alias, target)
    if "unicode" not in np.__dict__:
        np.unicode = np.str_


def build_sam_3d_body_estimator(args):
    """Initialize estimator with the same component wiring as sam-3d-body/demo.py."""

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=args.mhr_path
    )

    detector_path = args.detector_path or os.environ.get(
        "SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get(
        "SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    human_detector, human_segmentor, fov_estimator = None, None, None

    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path)

    # TODO: Fix Segmentor init
    if args.segmentor_name:
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )

    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    return estimator


def main():
    parser = argparse.ArgumentParser(
        description="Folder-based N-view MHR2SMPL inference")
    parser.add_argument("--input_dir", required=True,
                        help="Folder containing multiview images")
    parser.add_argument(
        "--image_glob",
        default="*",
        help="glob pattern inside input_dir (default: *)",
    )
    parser.add_argument(
        "--intrinsics_dir",
        default=None,
        help="Optional folder with per-image intrinsics named <image_stem>.json/.npy/.npz",
    )
    parser.add_argument(
        "--uncalibrated",
        action="store_true",
        help="synthesize intrinsics for all views (ignore intrinsics_dir)",
    )
    parser.add_argument("--bbox", default=None,
                        help="single bbox x1,y1,x2,y2 applied to all images (optional)")
    parser.add_argument("--focal", type=float,
                        default=None, help="override fx=fy for all views (pixels)")
    parser.add_argument(
        "--default_fov_deg",
        type=float,
        default=55.0,
        help="fallback horizontal FOV for synthesized intrinsics",
    )
    parser.add_argument(
        "--checkpoint_path",
        default='./checkpoints/sam-3d-body-dinov3/model.ckpt',
        help="SAM-3D-Body local checkpoint dir",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam3",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        help="path to MoHR assets (or set SAM3D_MHR_PATH)",
    )

    parser.add_argument(
        "--mv_model_path",
        default=str(MHR2SMPL_DIR /
                    "experiments/multiview_n30000_e500/best_model.pth"),
        help="multi-view model checkpoint path",
    )
    parser.add_argument(
        "--mapping_path",
        default=str(MHR2SMPL_DIR / "data/mhr2smpl_mapping.npz"),
        help="MHR->SMPL mapping npz",
    )
    parser.add_argument(
        "--sample_idx_path",
        default=str(MHR2SMPL_DIR / "data/smpl_vert_sample_indices.npy"),
        help="SMPL vertex sample indices",
    )
    parser.add_argument(
        "--smpl_model_path",
        default=str(MHR2SMPL_DIR / "data/SMPL_NEUTRAL_chumpy_free.pkl"),
        help="SMPL_NEUTRAL.pkl path",
    )
    parser.add_argument(
        "--smoother_dir",
        default=str(MHR2SMPL_DIR /
                    "experiments/smoother_w5"),
        help="optional SmootherMLP directory",
    )
    parser.add_argument(
        "--use_smoother",
        action="store_true",
        help="apply SmootherMLP via infer_smpl_joints (joint denoising)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="inference device",
    )
    parser.add_argument(
        "--output_dir",
        default=str(MHR2SMPL_DIR / "output_multi_images"),
        help="output directory",
    )
    parser.add_argument(
        "--save_mhr_mesh",
        action="store_true",
        help="save per-view SAM-3D-Body MHR meshes as OBJ",
    )
    parser.add_argument(
        "--skip_smpl_mesh",
        action="store_true",
        help="skip final SMPL OBJ export (still saves result npz unless disabled externally)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(args.input_dir, args.image_glob)
    print(f"Found {len(image_paths)} input images in {args.input_dir}")
    if len(image_paths) < 2:
        print("  Warning: fewer than 2 views found; fusion falls back to single-view behavior.")

    bbox_all = parse_bbox(args.bbox)

    print("[1/4] Loading SAM-3D-Body estimator...")
    estimator = build_sam_3d_body_estimator(args)
    print("  SAM-3D-Body estimator initialized.")
    # mhr_faces = np.asarray(estimator.faces, dtype=np.int32)

    # print(f"[2/4] Running Stage-1 on {len(image_paths)} images...")
    # hand_box_source = (
    #     "yolo_pose" if args.detector_name == "yolo_pose" else "body_decoder"
    # )
    # views = []
    # stage1_records = []
    # mhr_mesh_records = []
    # for i, img_path in enumerate(image_paths):
    #     img = cv2.imread(str(img_path))
    #     if img is None:
    #         raise FileNotFoundError(f"failed to read image: {img_path}")
    #     h, w = img.shape[:2]

    #     intr_path = None if args.uncalibrated else find_intrinsics_for_image(
    #         img_path, args.intrinsics_dir
    #     )
    #     cam_int = load_cam_int(intr_path)
    #     if cam_int is None:
    #         focal = (
    #             float(args.focal)
    #             if args.focal is not None
    #             else estimate_focal_from_fov(w, h, args.default_fov_deg)
    #         )
    #         cam_int = build_default_cam_int(w, h, focal)
    #         mode = "manual focal" if args.focal is not None else "estimated FOV"
    #         print(
    #             f"  [{i}] {img_path.name}: synthesized intrinsics ({mode}), fx=fy={focal:.1f}")
    #         intr_src = f"synthesized:{mode}"
    #     else:
    #         print(
    #             f"  [{i}] {img_path.name}: loaded intrinsics from {Path(intr_path).name}")
    #         intr_src = str(intr_path)

    #     pred = infer_single_view(
    #         estimator, str(img_path), bbox_all, cam_int, hand_box_source
    #     )
    #     pred_vertices = np.asarray(pred["pred_vertices"], dtype=np.float32)
    #     pred_cam_t = np.asarray(pred["pred_cam_t"], dtype=np.float32)
    #     views.append((pred_vertices, pred_cam_t))

    #     stage1_name = f"stage1_{i:03d}_{img_path.stem}.npz"
    #     np.savez(
    #         out_dir / stage1_name,
    #         image_path=str(img_path),
    #         cam_int=cam_int,
    #         intrinsics_source=intr_src,
    #         pred_vertices=pred_vertices,
    #         pred_cam_t=pred_cam_t,
    #     )
    #     stage1_records.append(stage1_name)

    #     if args.save_mhr_mesh:
    #         mhr_obj_name = f"mhr_mesh_{i:03d}_{img_path.stem}.obj"
    #         save_obj(out_dir / mhr_obj_name, pred_vertices, mhr_faces)
    #         mhr_mesh_records.append(mhr_obj_name)

    # print("[3/4] Multi-view fusion (MHR2SMPL)...")
    # mv_model = MHR2SMPLMultiView(
    #     model_path=args.mv_model_path,
    #     mapping_path=args.mapping_path,
    #     sample_idx_path=args.sample_idx_path,
    #     device=args.device,
    #     smoother_dir=args.smoother_dir,
    # )
    # smoothed_joints = None
    # if args.use_smoother:
    #     if args.smoother_dir is None:
    #         raise ValueError("--use_smoother requires --smoother_dir")
    #     go, body_pose, betas, weights, smoothed_joints = mv_model.infer_smpl_joints(
    #         views, smpl_model_path=args.smpl_model_path
    #     )
    # else:
    #     go, body_pose, betas, weights = mv_model.infer(views)

    # print("[4/4] Building canonical SMPL mesh...")
    # patch_chumpy_compat()
    # import smplx

    # smpl = smplx.SMPL(model_path=args.smpl_model_path,
    #                   gender="neutral").to(args.device)
    # smpl.eval()
    # for p in smpl.parameters():
    #     p.requires_grad_(False)

    # body_pose_69 = np.zeros(69, dtype=np.float32)
    # body_pose_69[:63] = body_pose.astype(np.float32)
    # with torch.no_grad():
    #     out = smpl(
    #         global_orient=torch.zeros(1, 3, device=args.device),
    #         body_pose=torch.from_numpy(
    #             body_pose_69).unsqueeze(0).to(args.device),
    #         betas=torch.from_numpy(betas.astype(
    #             np.float32)).unsqueeze(0).to(args.device),
    #     )
    # vertices = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    # joints = out.joints[0, :24].detach().cpu().numpy().astype(np.float32)
    # joints -= joints[0:1]
    # if smoothed_joints is not None:
    #     joints = np.asarray(smoothed_joints, dtype=np.float32)
    # faces = np.asarray(smpl.faces, dtype=np.int32)

    # if not args.skip_smpl_mesh:
    #     save_obj(out_dir / "smpl_mesh.obj", vertices, faces)
    # np.savez(
    #     out_dir / "result_multi_view.npz",
    #     go=go.astype(np.float32),
    #     body_pose=body_pose.astype(np.float32),
    #     betas=betas.astype(np.float32),
    #     view_weights=weights.astype(np.float32),
    #     image_paths=np.asarray([str(p) for p in image_paths]),
    #     joints_root_relative=joints,
    #     vertices=vertices,
    #     faces=faces,
    # )

    # print("")
    # print("Done.")
    # print(f"  Output dir: {out_dir}")
    # print(f"  Views processed: {len(image_paths)}")
    # print(f"  Stage1 files: {len(stage1_records)} saved (stage1_*.npz)")
    # if args.save_mhr_mesh:
    #     print(f"  MHR meshes: {len(mhr_mesh_records)} saved (mhr_mesh_*.obj)")
    # print(f"  Result npz:  {out_dir / 'result_multi_view.npz'}")
    # if args.skip_smpl_mesh:
    #     print("  Mesh obj:    skipped (--skip_smpl_mesh)")
    # else:
    #     print(f"  Mesh obj:    {out_dir / 'smpl_mesh.obj'}")
    # weight_text = ", ".join([f"{float(w):.3f}" for w in weights])
    # print(f"  View weights: [{weight_text}]")
    # if args.use_smoother:
    #     print("  Smoother: enabled (joint denoising applied).")
    #     print("  Note: for still images, temporal smoothing has limited effect.")
    # print("  Note: exported mesh is canonical/root-relative (global_orient set to zero).")


if __name__ == "__main__":
    main()
