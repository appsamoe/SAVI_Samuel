#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path


# Adjust these values and rerun to compare one set of thresholds across images.
H_MIN = 20
H_MAX = 60
S_MIN = 100
S_MAX = 255
V_MIN = 60
V_MAX = 255

IMAGE_NAMES = (
    "dog_2.jpg",
    "person_1.jpg",
    "person_3.jpg",
    "dog_1.jpg",
    "dog_3.jpg",
    "dog_4.jpg",
    "person_2.jpg",
    "person_4.jpg",
)


def make_grass_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    mask_h = np.logical_and(h > H_MIN, h < H_MAX)
    mask_s = np.logical_and(s > S_MIN, s < S_MAX)
    mask_v = np.logical_and(v > V_MIN, v < V_MAX)
    mask = np.logical_and(np.logical_and(mask_h, mask_s), mask_v)

    return mask.astype(np.uint8) * 255


def main():
    images_directory = Path(__file__).resolve().parent / "images"
    print(
        "HSV thresholds:",
        f"H=({H_MIN}, {H_MAX}),",
        f"S=({S_MIN}, {S_MAX}),",
        f"V=({V_MIN}, {V_MAX})",
    )
    print("Press any key to view the next image, or q/Esc to quit.")

    for image_name in IMAGE_NAMES:
        image_path = images_directory / image_name
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        height, width = image.shape[:2]
        display_image = cv2.resize(
            image, (round(width / 2), round(height / 2))
        )
        grass_mask = make_grass_mask(display_image)

        print(f"Showing {image_name}")
        cv2.imshow("Image", display_image)
        cv2.imshow("HSV grass mask", grass_mask)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
