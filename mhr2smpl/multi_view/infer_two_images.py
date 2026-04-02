#!/usr/bin/env python3
"""
Run multi-view MHR2SMPL inference from exactly two images.

Pipeline:
  image0,image1
    -> SAM-3D-Body Stage1 per image (pred_vertices, pred_cam_t)
    -> MHR2SMPL multi-view fusion
    -> SMPL canonical mesh export (.obj) + parameters (.npz)

Example:
  python mhr2smpl/multi_view/infer_two_images.py \
      --image0 /path/to/cam0.jpg \
      --image1 /path/to/cam1.jpg \
      --intrinsics0 /path/to/cam0_intrinsics.json \
      --intrinsics1 /path/to/cam1_intrinsics.json \
      --output_dir mhr2smpl/output_two_images
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

from notebook.utils import setup_sam_3d_body
import argparse
import inspect
import json
import os
import sys
from pathlib import Path

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
    if detector not in ("yolo_pose", "yolo"):
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


def main():
    parser = argparse.ArgumentParser(
        description="Two-image multi-view MHR2SMPL inference")
    parser.add_argument("--image0", required=True,
                        help="Path to camera-0 image")
    parser.add_argument("--image1", required=True,
                        help="Path to camera-1 image")
    parser.add_argument("--intrinsics0", default=None,
                        help="cam0 intrinsics (.json/.npy/.npz)")
    parser.add_argument("--intrinsics1", default=None,
                        help="cam1 intrinsics (.json/.npy/.npz)")
    parser.add_argument("--bbox0", default=None,
                        help="cam0 bbox as x1,y1,x2,y2 (optional)")
    parser.add_argument("--bbox1", default=None,
                        help="cam1 bbox as x1,y1,x2,y2 (optional)")
    parser.add_argument("--focal0", type=float,
                        default=1000.0, help="default fx=fy for cam0")
    parser.add_argument("--focal1", type=float,
                        default=1000.0, help="default fx=fy for cam1")
    parser.add_argument(
        "--local_checkpoint",
        default=str(PROJECT_DIR / "checkpoints/sam-3d-body-dinov3"),
        help="SAM-3D-Body local checkpoint dir",
    )
    parser.add_argument(
        "--detector_model",
        default=str(PROJECT_DIR / "checkpoints/yolo/yolo11m-pose.engine"),
        help="YOLO detector model path",
    )
    parser.add_argument(
        "--detector",
        default="yolo_pose",
        choices=["yolo_pose", "yolo", "vitdet", "none"],
        help="human detector backend",
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
        default=str(MHR2SMPL_DIR / "output_two_images"),
        help="output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img0 = cv2.imread(args.image0)
    img1 = cv2.imread(args.image1)
    if img0 is None:
        raise FileNotFoundError(f"failed to read image0: {args.image0}")
    if img1 is None:
        raise FileNotFoundError(f"failed to read image1: {args.image1}")

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    cam_int0 = load_cam_int(args.intrinsics0)
    cam_int1 = load_cam_int(args.intrinsics1)
    if cam_int0 is None:
        cam_int0 = build_default_cam_int(w0, h0, args.focal0)
    if cam_int1 is None:
        cam_int1 = build_default_cam_int(w1, h1, args.focal1)

    bbox0 = parse_bbox(args.bbox0)
    bbox1 = parse_bbox(args.bbox1)

    detector_name = "" if args.detector == "none" else args.detector
    detector_model = args.detector_model
    if args.detector in ("yolo_pose", "yolo"):
        detector_model = resolve_detector_model_path(
            args.detector, args.detector_model)
        if detector_model is None:
            raise FileNotFoundError(
                "No YOLO detector weights found.\n"
                "Tried --detector_model and common defaults under checkpoints/yolo/.\n"
                "Options:\n"
                "  1) Provide valid weights with --detector_model\n"
                "  2) Use --detector none and pass --bbox0/--bbox1 manually"
            )
    if args.detector == "none" and (bbox0 is None or bbox1 is None):
        raise ValueError(
            "--detector none requires both --bbox0 and --bbox1 "
            "(format: x1,y1,x2,y2)."
        )

    print("[1/4] Loading SAM-3D-Body estimator...")
    estimator = setup_sam_3d_body(
        local_checkpoint_path=args.local_checkpoint,
        detector_name=detector_name,
        detector_model=detector_model or "",
        fov_name=None,
        device=args.device,
    )

    print("[2/4] Running Stage-1 on both images...")
    hand_box_source = "yolo_pose" if args.detector == "yolo_pose" else "body_decoder"
    pred0 = infer_single_view(estimator, args.image0,
                              bbox0, cam_int0, hand_box_source)
    pred1 = infer_single_view(estimator, args.image1,
                              bbox1, cam_int1, hand_box_source)

    np.savez(
        out_dir / "stage1_cam0.npz",
        image_path=args.image0,
        cam_int=cam_int0,
        pred_vertices=pred0["pred_vertices"],
        pred_cam_t=pred0["pred_cam_t"],
    )
    np.savez(
        out_dir / "stage1_cam1.npz",
        image_path=args.image1,
        cam_int=cam_int1,
        pred_vertices=pred1["pred_vertices"],
        pred_cam_t=pred1["pred_cam_t"],
    )

    print("[3/4] Multi-view fusion (MHR2SMPL)...")
    mv_model = MHR2SMPLMultiView(
        model_path=args.mv_model_path,
        mapping_path=args.mapping_path,
        sample_idx_path=args.sample_idx_path,
        device=args.device,
        smoother_dir=args.smoother_dir,
    )
    views = [
        (np.asarray(pred0["pred_vertices"], dtype=np.float32),
         np.asarray(pred0["pred_cam_t"], dtype=np.float32)),
        (np.asarray(pred1["pred_vertices"], dtype=np.float32),
         np.asarray(pred1["pred_cam_t"], dtype=np.float32)),
    ]
    smoothed_joints = None
    if args.use_smoother:
        if args.smoother_dir is None:
            raise ValueError("--use_smoother requires --smoother_dir")
        go, body_pose, betas, weights, smoothed_joints = mv_model.infer_smpl_joints(
            views, smpl_model_path=args.smpl_model_path
        )
    else:
        go, body_pose, betas, weights = mv_model.infer(views)

    print("[4/4] Building canonical SMPL mesh...")
    patch_chumpy_compat()
    import smplx

    smpl = smplx.SMPL(model_path=args.smpl_model_path,
                      gender="neutral").to(args.device)
    smpl.eval()
    for p in smpl.parameters():
        p.requires_grad_(False)

    body_pose_69 = np.zeros(69, dtype=np.float32)
    body_pose_69[:63] = body_pose.astype(np.float32)
    with torch.no_grad():
        out = smpl(
            global_orient=torch.zeros(1, 3, device=args.device),
            body_pose=torch.from_numpy(
                body_pose_69).unsqueeze(0).to(args.device),
            betas=torch.from_numpy(betas.astype(
                np.float32)).unsqueeze(0).to(args.device),
        )
    vertices = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0, :24].detach().cpu().numpy().astype(np.float32)
    joints -= joints[0:1]
    if smoothed_joints is not None:
        joints = np.asarray(smoothed_joints, dtype=np.float32)
    faces = np.asarray(smpl.faces, dtype=np.int32)

    save_obj(out_dir / "smpl_mesh.obj", vertices, faces)
    np.savez(
        out_dir / "result_two_view.npz",
        go=go.astype(np.float32),
        body_pose=body_pose.astype(np.float32),
        betas=betas.astype(np.float32),
        view_weights=weights.astype(np.float32),
        joints_root_relative=joints,
        vertices=vertices,
        faces=faces,
    )

    print("")
    print("Done.")
    print(f"  Output dir: {out_dir}")
    print(f"  Stage1 cam0: {out_dir / 'stage1_cam0.npz'}")
    print(f"  Stage1 cam1: {out_dir / 'stage1_cam1.npz'}")
    print(f"  Result npz:  {out_dir / 'result_two_view.npz'}")
    print(f"  Mesh obj:    {out_dir / 'smpl_mesh.obj'}")
    print(f"  View weights: [{weights[0]:.3f}, {weights[1]:.3f}]")
    if args.use_smoother:
        print("  Smoother: enabled (joint denoising applied).")
        print("  Note: for only two still images, temporal smoothing has limited effect.")
    print("  Note: exported mesh is canonical/root-relative (global_orient set to zero).")


if __name__ == "__main__":
    main()
