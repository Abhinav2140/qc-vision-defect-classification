"""
inference_pipeline.py — Real-time inspection loop.

Wires together: camera frame capture (OpenCV) -> preprocessing -> model
inference -> severity/decision logic -> reject actuator -> analytics DB.

This is written to run on an edge box next to the line (Jetson Orin /
industrial PC with GPU) rather than round-tripping frames to the cloud —
at 15-60 fps and sub-100ms decision budgets, network round trips are
usually not viable, and most manufacturers don't want production images
leaving the plant network anyway.

Usage:
    python inference_pipeline.py --camera 0 --checkpoint qc_model.pt \
        --machine_id M-04 --batch_id B-2291 --shift Morning

Swap `--camera 0` for an RTSP URL to use an industrial GigE/IP camera:
    --camera rtsp://192.168.1.50:554/stream1
"""

import argparse
import time
import uuid

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from model import build_model, DEFECT_CLASSES
from dataset import build_transforms
from severity import decide, DEFAULT_REJECT_THRESHOLDS
from reject_actuator import SimulatedActuator
from database import Database


class InspectionPipeline:
    def __init__(self, checkpoint_path, machine_id, batch_id, shift,
                 camera_id="CAM-1", device=None, actuator=None, db_path="qc_inspections.db"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model = build_model({"backbone": ckpt.get("backbone", "efficientnet_b0")})
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = build_transforms(train=False)
        self.machine_id, self.batch_id, self.shift, self.camera_id = (
            machine_id, batch_id, shift, camera_id
        )
        self.actuator = actuator or SimulatedActuator()
        self.db = Database(db_path)

    @torch.no_grad()
    def infer_frame(self, frame_bgr):
        """frame_bgr: a single OpenCV BGR frame. Returns a Decision + timing."""
        t0 = time.time()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        out = self.model(tensor)
        probs = F.softmax(out["defect_logits"], dim=1)[0]
        conf, pred_idx = probs.max(dim=0)
        defect_type = DEFECT_CLASSES[pred_idx.item()]
        severity = out["severity"][0].item()
        bbox = out["bbox"][0].tolist() if "bbox" in out else None

        decision = decide(defect_type, severity, conf.item(), DEFAULT_REJECT_THRESHOLDS)
        inference_ms = (time.time() - t0) * 1000
        return decision, bbox, inference_ms

    def process_and_act(self, frame_bgr, part_id=None):
        part_id = part_id or str(uuid.uuid4())[:8]
        decision, bbox, inference_ms = self.infer_frame(frame_bgr)

        if decision.action == "reject":
            self.actuator.trigger_reject(part_id, decision.defect_type, decision.severity)

        self.db.log_inspection(
            part_id=part_id, shift=self.shift, machine_id=self.machine_id,
            batch_id=self.batch_id, camera_id=self.camera_id,
            defect_type=decision.defect_type, severity=decision.severity,
            confidence=decision.confidence, band=decision.band,
            action=decision.action, inference_ms=inference_ms,
        )
        return decision, bbox, inference_ms

    def run_camera_loop(self, camera_source, max_frames=None, show_preview=False):
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {camera_source}")

        frame_count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                decision, bbox, inference_ms = self.process_and_act(frame)

                if decision.action != "pass":
                    print(f"[{decision.action.upper()}] {decision.defect_type} "
                          f"severity={decision.severity:.2f} conf={decision.confidence:.2f} "
                          f"({inference_ms:.1f} ms)")

                if show_preview:
                    label = f"{decision.defect_type} ({decision.action})"
                    color = (0, 0, 255) if decision.action == "reject" else (0, 200, 0)
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, color, 2)
                    cv2.imshow("QC Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="camera index or RTSP URL")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--machine_id", required=True)
    ap.add_argument("--batch_id", required=True)
    ap.add_argument("--shift", required=True, choices=["Morning", "Afternoon", "Night"])
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    camera_source = int(args.camera) if args.camera.isdigit() else args.camera

    pipeline = InspectionPipeline(
        checkpoint_path=args.checkpoint, machine_id=args.machine_id,
        batch_id=args.batch_id, shift=args.shift,
    )
    pipeline.run_camera_loop(camera_source, show_preview=args.preview)


if __name__ == "__main__":
    main()
