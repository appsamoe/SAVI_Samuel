#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path


def make_grass_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    mask_h = np.logical_and(h > 20, h < 60)
    mask_s = np.logical_and(s > 100, s < 255)
    mask_v = np.logical_and(v > 60, v < 255)

    return np.logical_and(np.logical_and(mask_h, mask_s), mask_v)


def main():
    image_path = Path(__file__).resolve().parent / "images" / "dog_2.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    image = cv2.resize(image, (round(width / 2), round(height / 2)))

    grass_mask = make_grass_mask(image).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)

    grass_mask = cv2.morphologyEx(
        grass_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    grass_mask = cv2.morphologyEx(
        grass_mask, cv2.MORPH_CLOSE, kernel, iterations=1
    )

    dog_mask = cv2.bitwise_not(grass_mask)
    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dog_mask, connectivity=8
    )

    cv2.imshow("Original", image)
    cv2.imshow("Cleaned grass mask", grass_mask)
    cv2.imshow("Inverted mask", dog_mask)

    if number_of_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x = stats[largest_label, cv2.CC_STAT_LEFT]
        y = stats[largest_label, cv2.CC_STAT_TOP]
        component_width = stats[largest_label, cv2.CC_STAT_WIDTH]
        component_height = stats[largest_label, cv2.CC_STAT_HEIGHT]

        largest_component = np.zeros_like(dog_mask)
        largest_component[labels == largest_label] = 255

        image_with_rectangle = image.copy()
        cv2.rectangle(
            image_with_rectangle,
            (x, y),
            (x + component_width - 1, y + component_height - 1),
            (0, 0, 255),
            2,
        )

        cv2.imshow("Largest component", largest_component)
        cv2.imshow("Largest component bounding box", image_with_rectangle)
    else:
        print("No foreground component was found in the inverted mask.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
