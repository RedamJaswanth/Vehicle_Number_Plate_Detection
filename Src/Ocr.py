import cv2
import easyocr


class PlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(
            ["en"],
            gpu=False
        )

    def read_plate(self, plate_image):
        if plate_image is None:
            return ""

        if plate_image.size == 0:
            return ""

        # Convert to grayscale
        gray = cv2.cvtColor(
            plate_image,
            cv2.COLOR_BGR2GRAY
        )

        # Resize for better OCR
        scale = 2
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        results = self.reader.readtext(
            gray,
            detail=0,
            paragraph=False
        )

        if not results:
            return ""

        text = " ".join(results)

        # Remove unwanted spaces
        text = text.replace(" ", "")
        text = text.replace("-", "")
        text = text.replace("_", "")

        return text.upper()