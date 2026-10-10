import cv2


def crop_image(frame, bbox):

    x1, y1, x2, y2 = bbox

    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def draw_vehicle(
    frame,
    bbox,
    label,
    confidence
):

    x1, y1, x2, y2 = bbox

    # Green vehicle box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # Vehicle label
    text = f"{label.upper()} {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    label_y1 = max(
        0,
        y1 - text_height - baseline - 8
    )

    label_y2 = y1

    label_x2 = min(
        frame.shape[1],
        x1 + text_width + 12
    )

    # Black background
    cv2.rectangle(
        frame,
        (x1, label_y1),
        (label_x2, label_y2),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x1 + 5, y1 - 6),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )


def draw_plate(
    frame,
    bbox,
    plate_text,
    confidence
):

    x1, y1, x2, y2 = bbox

    # Red plate box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    if plate_text:

        text = plate_text.upper()

    else:

        text = "PLATE"


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_y = y2 + text_height + 8

    # If there isn't room below the plate,
    # put text above it.
    if text_y > frame.shape[0]:

        text_y = max(
            y1 - 8,
            text_height + 5
        )

    text_x = x1

    text_x2 = min(
        frame.shape[1],
        text_x + text_width + 12
    )

    text_y1 = max(
        0,
        text_y - text_height - 8
    )

    text_y2 = min(
        frame.shape[0],
        text_y + 4
    )

    # Black background for readability
    cv2.rectangle(
        frame,
        (text_x, text_y1),
        (text_x2, text_y2),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (text_x + 5, text_y),
        font,
        font_scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA
    )


def put_title(frame, text):

    cv2.putText(
        frame,
        text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )