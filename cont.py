import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not found. Install with: pip install ultralytics", file=sys.stderr)
    raise

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle counting on a local video using YOLOv8 + ByteTrack.")
    parser.add_argument("--video", required=True, help="rodoviaComZoomDireita.mp4")
    parser.add_argument("--output", default="contadoZoomDireita.mp4", help="Path to save annotated output video (MP4).")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO weights path or name (e.g., yolov8n.pt).")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--classes", default="car,motorcycle,bus,truck",
                        help="Comma-separated class names to count (must exist in model.names).")
    parser.add_argument("--line", nargs=4, type=int, metavar=("x1","y1","x2","y2"),
                        help="Counting line as 4 integers in pixels. If omitted, a horizontal line through the middle is used.")
    parser.add_argument("--show", action="store_true", help="Show a realtime preview window.")
    parser.add_argument("--device", default=None, help="Set device, e.g., 'cpu', '0' (GPU 0). Default: auto.")
    return parser.parse_args()

def compute_side(px, py, x1, y1, x2, y2):
    # Returns sign of point relative to directed line (x1,y1)->(x2,y2)
    return np.sign((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))

def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Load model
    model = YOLO(args.weights)

    # Resolve class names to IDs
    model_names = model.names  # dict: id -> name
    name_to_id = {v: k for k, v in model_names.items()}

    wanted_names = [n.strip() for n in args.classes.split(",") if n.strip()]
    missing = [n for n in wanted_names if n not in name_to_id]
    if missing:
        print(f"WARNING: These classes are not in the model and will be ignored: {missing}", file=sys.stderr)
    class_ids = sorted({name_to_id[n] for n in wanted_names if n in name_to_id})

    # Open video to get properties
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if args.line is None:
        # Default: horizontal line across the middle of the frame
        x1, y1, x2, y2 = 0, height // 5, width, height // 5
    else:
        x1, y1, x2, y2 = args.line

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: cannot open writer for: {args.output}", file=sys.stderr)
        sys.exit(1)

    # Tracking / Counting state
    prev_pos = {}            # id -> (x,y)
    counted_ids = set()      # ids already counted on line crossing
    total_count = 0
    per_class_count = defaultdict(int)

    # Run tracker on video stream
    results_generator = model.track(
        source=args.video,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=args.conf,
        iou=args.iou,
        classes=class_ids if class_ids else None,
        device=args.device,
        verbose=False,
    )

    # Colors for drawing (BGR)
    line_color = (0, 255, 255)
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    def draw_hud(img):
        # Draw counting line
        cv2.line(img, (x1, y1), (x2, y2), line_color, 2)
        # HUD box
        hud = f"Total: {total_count}  |  " + "  ".join([f"{model_names[k]}: {v}" for k, v in sorted(per_class_count.items())])
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 10), bg_color, -1)
        cv2.putText(img, hud, (16, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    for r in results_generator:
        frame = r.plot()

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()

            for obj_id, cls_id, box in zip(ids, clss, xyxy):
                x1b, y1b, x2b, y2b = box[:4]
                cx = int((x1b + x2b) / 2)
                cy = int((y1b + y2b) / 2)

                # posição atual em relação à linha
                curr_side = compute_side(cx, cy, x1, y1, x2, y2)
                prev = prev_pos.get(obj_id, None)

                if prev is not None:
                    prev_cx, prev_cy = prev
                    prev_side = compute_side(prev_cx, prev_cy, x1, y1, x2, y2)

                    # veículo cruzou a linha → conta por ID
                    if prev_side != 0 and curr_side != 0 and np.sign(prev_side) != np.sign(curr_side):
                        if obj_id not in counted_ids:
                            counted_ids.add(obj_id)
                            total_count += 1
                            per_class_count[cls_id] += 1

                # sempre atualiza a posição anterior
                prev_pos[obj_id] = (cx, cy)

        draw_hud(frame)

        if args.show:
            cv2.imshow("Vehicle Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        writer.write(frame)

    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # Summary
    print("\n=== Counting Summary ===")
    print(f"Total vehicles: {counted_ids}")
    for cls_id in sorted(per_class_count.keys()):
        print(f"{model_names[cls_id]}: {per_class_count[cls_id]}")
    print(f"\nAnnotated video saved to: {args.output}")

if __name__ == "__main__":
    main()


#python cont.py --video rodoviaEditado.mp4 --show