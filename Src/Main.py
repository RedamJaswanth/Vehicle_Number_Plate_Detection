import cv2
import os

from Src.Detector import VehicleDetector
from Src.Plate_Detector import PlateDetector
from Src.Ocr import PlateOCR
from Utils.Helpers import (
    crop_image,
    draw_vehicle,
    draw_plate
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_VIDEO = "Data/Input/Sample.mp4"

OUTPUT_VIDEO = "Data/Output/result.mp4"
OUTPUT_TEXT = "Data/Output/result.txt"

# Detection frequency
VEHICLE_DETECT_EVERY = 4
PLATE_DETECT_EVERY = 5
OCR_EVERY = 15

# Display size
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    "Data/Output",
    exist_ok=True
)


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("Loading Vehicle YOLO...")

vehicle_detector = VehicleDetector()

print("Vehicle model loaded.")


print()
print("Loading License Plate YOLO...")

plate_detector = PlateDetector()

print("License plate model loaded.")


print()
print("Loading EasyOCR...")

ocr = PlateOCR()

print("EasyOCR loaded.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

if fps <= 0:
    fps = 30.0


print()
print("==========================================")
print("VIDEO INFORMATION")
print("==========================================")

print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")

print("==========================================")


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


if not out.isOpened():

    print(
        "ERROR: Could not create output video."
    )

    cap.release()

    exit()


# ============================================================
# CACHE
# ============================================================

cached_vehicles = []
cached_plates = []


# ============================================================
# RESULT STORAGE
# ============================================================

detected_results = []


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_vehicle_result(
    vehicle_name,
    plate_text,
    confidence,
    frame_number,
    video_fps
):

    vehicle_name = vehicle_name.upper()

    if plate_text:
        plate_text = plate_text.upper()
    else:
        plate_text = "NOT DETECTED"


    # ========================================================
    # CHECK IF SAME PLATE ALREADY EXISTS
    # ========================================================

    if plate_text != "NOT DETECTED":

        for result in detected_results:

            if result["plate"] == plate_text:

                return


    # ========================================================
    # ADD NEW RESULT
    # ========================================================

    timestamp = frame_number / video_fps

    minutes = int(timestamp // 60)

    seconds = int(timestamp % 60)


    detected_results.append({

        "vehicle": vehicle_name,

        "plate": plate_text,

        "confidence": confidence,

        "frame": frame_number,

        "timestamp": f"{minutes:02d}:{seconds:02d}"

    })


# ============================================================
# FRAME COUNTER
# ============================================================

frame_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_count += 1


    # ========================================================
    # VEHICLE DETECTION
    # ========================================================

    if (
        frame_count == 1
        or frame_count % VEHICLE_DETECT_EVERY == 0
    ):

        cached_vehicles = (
            vehicle_detector.detect(frame)
        )


    # ========================================================
    # PLATE DETECTION
    # ========================================================

    if (
        frame_count == 1
        or frame_count % PLATE_DETECT_EVERY == 0
    ):

        new_plates = (
            plate_detector.detect(frame)
        )


        # Preserve old OCR results
        for new_plate in new_plates:

            new_plate["text"] = ""

            new_x1, new_y1, new_x2, new_y2 = (
                new_plate["bbox"]
            )

            new_center_x = (
                new_x1 + new_x2
            ) // 2

            new_center_y = (
                new_y1 + new_y2
            ) // 2


            best_distance = float("inf")

            best_old_plate = None


            for old_plate in cached_plates:

                old_x1, old_y1, old_x2, old_y2 = (
                    old_plate["bbox"]
                )

                old_center_x = (
                    old_x1 + old_x2
                ) // 2

                old_center_y = (
                    old_y1 + old_y2
                ) // 2


                distance = (
                    (new_center_x - old_center_x) ** 2
                    +
                    (new_center_y - old_center_y) ** 2
                )


                if distance < best_distance:

                    best_distance = distance

                    best_old_plate = old_plate


            # Preserve OCR text
            if (
                best_old_plate is not None
                and best_distance < 10000
            ):

                new_plate["text"] = (
                    best_old_plate.get(
                        "text",
                        ""
                    )
                )


        cached_plates = new_plates


    # ========================================================
    # OCR
    # ========================================================

    if (
        frame_count == 1
        or frame_count % OCR_EVERY == 0
    ):

        for plate in cached_plates:

            plate_image = crop_image(
                frame,
                plate["bbox"]
            )


            if plate_image is None:
                continue


            plate_height, plate_width = (
                plate_image.shape[:2]
            )


            # Ignore very small plates
            if (
                plate_width < 25
                or plate_height < 8
            ):
                continue


            try:

                plate_text = ocr.read_plate(
                    plate_image
                )


                if plate_text:

                    plate["text"] = plate_text


            except Exception as error:

                print(
                    f"OCR error: {error}"
                )


    # ========================================================
    # DRAW VEHICLES
    # ========================================================

    for vehicle in cached_vehicles:

        vehicle_bbox = vehicle["bbox"]

        vx1, vy1, vx2, vy2 = (
            vehicle_bbox
        )

        vehicle_name = (
            vehicle["class_name"]
        )

        vehicle_confidence = (
            vehicle["confidence"]
        )


        # ====================================================
        # GREEN VEHICLE BOX
        # ====================================================

        draw_vehicle(
            frame,
            vehicle_bbox,
            vehicle_name,
            vehicle_confidence
        )


        # ====================================================
        # FIND PLATE INSIDE VEHICLE
        # ====================================================

        matched_plate = None

        best_distance = float("inf")


        for plate in cached_plates:

            px1, py1, px2, py2 = (
                plate["bbox"]
            )


            plate_center_x = (
                px1 + px2
            ) // 2

            plate_center_y = (
                py1 + py2
            ) // 2


            # Plate center must be inside vehicle
            if (
                vx1 <= plate_center_x <= vx2
                and
                vy1 <= plate_center_y <= vy2
            ):

                vehicle_center_x = (
                    vx1 + vx2
                ) // 2

                vehicle_center_y = (
                    vy1 + vy2
                ) // 2


                distance = (
                    (plate_center_x - vehicle_center_x) ** 2
                    +
                    (plate_center_y - vehicle_center_y) ** 2
                )


                if distance < best_distance:

                    best_distance = distance

                    matched_plate = plate


        # ====================================================
        # VEHICLE INFORMATION
        # ====================================================

        if matched_plate is not None:

            plate_text = (
                matched_plate.get(
                    "text",
                    ""
                )
            )


            if plate_text:

                information = (
                    f"{vehicle_name.upper()}  |  "
                    f"PLATE: {plate_text}"
                )

            else:

                information = (
                    f"{vehicle_name.upper()}  |  "
                    f"PLATE: READING..."
                )

        else:

            plate_text = ""

            information = (
                f"{vehicle_name.upper()}  |  "
                f"PLATE: NOT DETECTED"
            )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        # Only save when OCR actually found a plate.
        # This prevents the same vehicle from creating
        # many "NOT DETECTED" entries.

        if plate_text:

            save_vehicle_result(
                vehicle_name,
                plate_text,
                vehicle_confidence,
                frame_count,
                fps
            )


        # ====================================================
        # LARGE INFORMATION TEXT
        # ====================================================

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.85

        thickness = 2


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            information,
            font,
            font_scale,
            thickness
        )


        text_x = max(
            5,
            vx1
        )


        # Put information below vehicle
        text_y = (
            vy2 +
            text_height +
            15
        )


        # If outside bottom, put above
        if text_y > height - 5:

            text_y = max(
                vy1 - 10,
                text_height + 10
            )


        # Black background
        bg_x1 = text_x

        bg_y1 = (
            text_y -
            text_height -
            10
        )

        bg_x2 = min(
            width,
            text_x +
            text_width +
            12
        )

        bg_y2 = min(
            height,
            text_y +
            5
        )


        cv2.rectangle(
            frame,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (0, 0, 0),
            -1
        )


        # White text
        cv2.putText(
            frame,
            information,
            (
                text_x + 5,
                text_y
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


    # ========================================================
    # DRAW PLATE BOXES
    # ========================================================

    for plate in cached_plates:

        draw_plate(
            frame,
            plate["bbox"],
            plate.get(
                "text",
                ""
            ),
            plate["confidence"]
        )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "VEHICLE & LICENSE PLATE DETECTION",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # SAVE VIDEO
    # ========================================================

    out.write(frame)


    # ========================================================
    # DISPLAY
    # ========================================================

    display_frame = cv2.resize(
        frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )
    )


    cv2.imshow(
        "Vehicle Number Plate Detection",
        display_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        print()
        print("Stopped by user.")

        break


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_count % 100 == 0:

        percentage = (
            frame_count /
            total_frames
        ) * 100

        print(
            f"Processing: "
            f"{frame_count}/{total_frames} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()

out.release()

cv2.destroyAllWindows()


# ============================================================
# SAVE RESULT.TXT
# ============================================================

with open(
    OUTPUT_TEXT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "========================================\n"
    )

    file.write(
        "VEHICLE NUMBER PLATE DETECTION RESULTS\n"
    )

    file.write(
        "========================================\n\n"
    )


    if detected_results:

        for index, result in enumerate(
            detected_results,
            start=1
        ):

            file.write(
                f"Vehicle {index}\n"
            )

            file.write(
                f"Vehicle Type : "
                f"{result['vehicle']}\n"
            )

            file.write(
                f"Plate Number : "
                f"{result['plate']}\n"
            )

            file.write(
                f"Confidence   : "
                f"{result['confidence']:.2f}\n"
            )

            file.write(
                f"Timestamp    : "
                f"{result['timestamp']}\n"
            )

            file.write(
                f"Frame        : "
                f"{result['frame']}\n"
            )

            file.write(
                "\n"
            )

    else:

        file.write(
            "No license plates were successfully "
            "recognized by OCR.\n\n"
        )


    file.write(
        "========================================\n"
    )

    file.write(
        f"TOTAL RECOGNIZED VEHICLES : "
        f"{len(detected_results)}\n"
    )

    file.write(
        "========================================\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("==========================================")
print("PROCESSING COMPLETED")
print("==========================================")

print(
    f"Video Output : {OUTPUT_VIDEO}"
)

print(
    f"Text Output  : {OUTPUT_TEXT}"
)

print(
    f"Recognized Vehicles : "
    f"{len(detected_results)}"
)

print(
    f"Original FPS : {fps:.2f}"
)

print(
    f"Output FPS   : {fps:.2f}"
)

print("==========================================")