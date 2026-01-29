import cv2
import numpy as np

from src.video_reader import VideoReader
from src.helmet_detector import HelmetDetector
from src.violation import HelmetViolationChecker
from src.plate_detector import PlateDetector
from src.ocr import PlateOCR
from src.report import HelmetReportManager

from sort.sort import Sort

# =========================
# CONFIG
# =========================
VIDEO_PATH = "input/video.mp4"

# =========================
# INIT MODULES
# =========================
reader = VideoReader(VIDEO_PATH)
detector = HelmetDetector()
violation_checker = HelmetViolationChecker()

plate_detector = PlateDetector("models/license_plate.pt")
ocr = PlateOCR()
report_manager = HelmetReportManager("reports")

person_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

# Store already violated IDs
violated_person_ids = set()

delay = int(1000 / reader.fps)

print("[INFO] Helmet violation system started")

# =========================
# MAIN LOOP
# =========================
while True:
    frame = reader.read()
    if frame is None:
        break

    persons, bikes, helmets = detector.detect(frame)

    # -----------------------
    # TRACK PERSONS
    # -----------------------
    dets = []
    for x1, y1, x2, y2, conf in persons:
        dets.append([x1, y1, x2, y2, conf])

    dets = np.array(dets) if len(dets) else np.empty((0, 5))
    tracked_persons = person_tracker.update(dets)

    # -----------------------
    # DRAW HELMETS
    # -----------------------
    for hx1, hy1, hx2, hy2, label, conf in helmets:
        color = (0, 255, 0) if label == "WITH_HELMET" else (0, 0, 255)
        text = "HELMET" if label == "WITH_HELMET" else "NO HELMET"

        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
        cv2.putText(
            frame, text,
            (hx1, hy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # -----------------------
    # DRAW BIKES
    # -----------------------
    for bx1, by1, bx2, by2, conf in bikes:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
        cv2.putText(
            frame, "BIKE",
            (bx1, by1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

    # -----------------------
    # CHECK VIOLATIONS (ID BASED)
    # -----------------------
    for p in tracked_persons:
        px1, py1, px2, py2, person_id = map(int, p)

        # Draw person
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"PERSON ID {person_id}",
            (px1, py1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        # Skip if already violated
        if person_id in violated_person_ids:
            continue

        violation = violation_checker.check_single(
            person_box=(px1, py1, px2, py2),
            bikes=bikes,
            helmets=helmets
        )

        if violation:
            violated_person_ids.add(person_id)

            print(f"[HELMET VIOLATION] Person ID={person_id}")

            bx1, by1, bx2, by2 = violation

            # -----------------------
            # CROP BIKE IMAGE
            # -----------------------
            bike_img = frame[by1:by2, bx1:bx2].copy()

            # -----------------------
            # PLATE DETECTION
            # -----------------------
            plate_img, plate_box = plate_detector.detect(bike_img)

            # -----------------------
            # OCR
            # -----------------------
            plate_text = ocr.read(plate_img) if plate_img is not None else ""

            # -----------------------
            # SAVE REPORT
            # -----------------------
            report_manager.create_report(
                bike_img=bike_img,
                plate_img=plate_img,
                plate_text=plate_text
            )

            # Draw violation
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                "NO HELMET VIOLATION",
                (bx1, by1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # -----------------------
    # DISPLAY
    # -----------------------
    cv2.imshow("Helmet Violation - Full Pipeline", reader.resize_for_display(frame))

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
reader.release()
print("[INFO] Helmet violation system stopped")
