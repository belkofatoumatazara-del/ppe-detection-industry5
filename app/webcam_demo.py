"""
Standalone live webcam PPE demo (OpenCV) - the reliable backup for a live demo.
Industry 5.0 worker safety monitoring (YOLOv8).

Run locally:
    pip install ultralytics opencv-python
    python app/webcam_demo.py

Press 'q' to quit. Put best.pt next to this script (or edit MODEL_PATH).
"""
import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CONF = 0.30          # lower = catches more violations (higher recall)

model = YOLO(MODEL_PATH)
names = model.names
cap = cv2.VideoCapture(0)   # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try a different index (1, 2).")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(frame, conf=CONF, verbose=False)[0]
    annotated = result.plot()

    # Compliance banner: any NO-* detection = violation
    violations = sorted({
        names[int(c)] for c in result.boxes.cls.tolist()
        if names[int(c)].upper().startswith("NO-")
    })
    if violations:
        text, color = "NON-COMPLIANT: " + ", ".join(violations), (0, 0, 255)
    else:
        text, color = "COMPLIANT", (0, 180, 0)

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(annotated, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("PPE Compliance - Industry 5.0  (press q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
