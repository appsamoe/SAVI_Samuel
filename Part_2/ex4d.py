#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path


# Set to True to reject match positions whose template center falls on grass.
USE_NON_GRASS_SEARCH_MASK = False


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def make_non_grass_mask(scene):
    hsv_image = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    grass = (
        (h > 20) & (h < 60) &
        (s > 100) & (s < 255) &
        (v > 60) & (v < 255)
    )
    return np.logical_not(grass)


def locate_template(scene, template):
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_height, template_width = template_gray.shape

    if (template_height > scene_gray.shape[0] or
            template_width > scene_gray.shape[1]):
        raise ValueError("The template must be smaller than the scene.")

    response = cv2.matchTemplate(
        scene_gray, template_gray, cv2.TM_CCOEFF_NORMED
    )

    if USE_NON_GRASS_SEARCH_MASK:
        non_grass = make_non_grass_mask(scene)
        valid_centers = non_grass[
            template_height // 2:template_height // 2 + response.shape[0],
            template_width // 2:template_width // 2 + response.shape[1],
        ]
        response[~valid_centers] = -1

    _, score, _, location = cv2.minMaxLoc(response)
    return location, template_width, template_height, score


def main():
    images_directory = Path(__file__).resolve().parent / "images"
    scene = load_image(images_directory / "cenario.jpg")
    template = load_image(images_directory / "modelo.png")

    location, width, height, score = locate_template(scene, template)
    x, y = location

    if score <= -1:
        raise RuntimeError("No valid template position remained to search.")

    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    highlighted = cv2.cvtColor(gray_scene, cv2.COLOR_GRAY2BGR)
    highlighted[y:y + height, x:x + width] = scene[y:y + height, x:x + width]
    cv2.rectangle(
        highlighted, (x, y), (x + width, y + height), (0, 0, 255), 2
    )

    print(f"Detection at {location}; match score={score:.3f}")
    if USE_NON_GRASS_SEARCH_MASK:
        print("Search candidates were restricted by the HSV non-grass mask.")
    cv2.imshow("Scene", scene)
    cv2.imshow("Dog highlighted; outside detection is grayscale", highlighted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
