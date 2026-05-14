import cv2
import torch
import argparse
import os
import sys
import time
from ultralytics import YOLO


def load_source(source):
    # Webcam
    if source == "0":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam.")
            sys.exit()
        return "video", cap

    # File
    if not os.path.exists(source):
        print(f"❌ File not found: {source}")
        sys.exit()

    img_formats = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
    if source.split('.')[-1].lower() in img_formats:
        img = cv2.imread(source)
        if img is None:
            print("❌ Failed to load image.")
            sys.exit()
        return "image", img

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Cannot open video file.")
        sys.exit()

    return "video", cap


def draw_boxes(frame, results, names, thickness):
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{names[cls_id]} {conf:.2f}"
            color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                        help="Image, video path or '0' for webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--save", action="store_true",
                        help="Save output")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Save directory")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    try:
        model = YOLO(args.model)
        model.to(device)
    except Exception as e:
        print("❌ Model loading failed:", e)
        sys.exit()

    mode, src = load_source(args.source)

    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)

    # ---------------- IMAGE ----------------
    if mode == "image":
        print("Processing image...")
        results = model(src, conf=args.conf)
        frame = draw_boxes(src, results, model.names, 2)

        cv2.imshow("Detection", frame)

        if args.save:
            out_path = os.path.join(args.save_dir, "result.jpg")
            cv2.imwrite(out_path, frame)
            print(f"✅ Saved at {out_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Done.")
        return

    # ---------------- VIDEO ----------------
    print("Processing video... Press 'q' to quit.")

    cap = src
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame, conf=args.conf)
        frame = draw_boxes(frame, results, model.names, 2)
        fps = 1 / (time.time() - start)

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        if args.save and writer is None:
            h, w = frame.shape[:2]
            out_path = os.path.join(args.save_dir, "result.mp4")
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (w, h)
            )
            print(f"Saving to {out_path}")

        if writer:
            writer.write(frame)

        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()
