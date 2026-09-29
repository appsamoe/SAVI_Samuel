#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path


SCALE_FACTORS = np.linspace(0.5, 2.0, 31)
MINIMUM_MATCH_SCORE = 0.4
MAX_DETECTIONS = 2


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def find_matches(scene_gray, template_gray):
    candidates = []

    for scale in SCALE_FACTORS:
        scaled_template = cv2.resize(
            template_gray,
            None,
            fx=float(scale),
            fy=float(scale),
            interpolation=(
                cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            ),
        )
        template_height, template_width = scaled_template.shape
        if (template_height > scene_gray.shape[0] or
                template_width > scene_gray.shape[1]):
            continue

        response = cv2.matchTemplate(
            scene_gray, scaled_template, cv2.TM_CCOEFF_NORMED
        )

        for _ in range(MAX_DETECTIONS):
            _, score, _, location = cv2.minMaxLoc(response)
            if score < MINIMUM_MATCH_SCORE:
                break

            x, y = location
            candidates.append(
                (score, x, y, template_width, template_height)
            )

            # Suppress nearby positions at this scale to expose another peak.
            left = max(0, x - template_width // 2)
            top = max(0, y - template_height // 2)
            right = min(response.shape[1], x + template_width // 2 + 1)
            bottom = min(response.shape[0], y + template_height // 2 + 1)
            response[top:bottom, left:right] = -1

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    detections = []

    for candidate in candidates:
        score, x, y, width, height = candidate
        overlaps_same_detection = False
        for _, kept_x, kept_y, kept_width, kept_height in detections:
            intersection_width = max(
                0, min(x + width, kept_x + kept_width) - max(x, kept_x)
            )
            intersection_height = max(
                0, min(y + height, kept_y + kept_height) - max(y, kept_y)
            )
            intersection = intersection_width * intersection_height
            union = width * height + kept_width * kept_height - intersection
            if union > 0 and intersection / union > 0.35:
                overlaps_same_detection = True
                break

        if not overlaps_same_detection:
            detections.append(candidate)
            if len(detections) == MAX_DETECTIONS:
                break

    return detections


def main():
    images_directory = Path(__file__).resolve().parent / "images"
    scene = load_image(images_directory / "cenario_2.jpg")

    print(
        "Drag a rectangle around one dog's head, then press Enter or Space. "
        "Press Esc to cancel."
    )
    x, y, width, height = cv2.selectROI(
        "Select a dog template", scene, showCrosshair=True
    )
    cv2.destroyWindow("Select a dog template")

    if width == 0 or height == 0:
        print("No template was selected.")
        return

    template = scene[y:y + height, x:x + width].copy()
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = scene.copy()
    detections = find_matches(scene_gray, template_gray)

    for index, (score, x, y, box_width, box_height) in enumerate(detections):
        color = (0, 0, 255) if index == 0 else (0, 255, 0)
        cv2.rectangle(
            result,
            (x, y),
            (x + box_width, y + box_height),
            color,
            2,
        )
        cv2.putText(
            result,
            f"match {index + 1}: {score:.3f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        print(f"Match {index + 1}: score={score:.3f}, box={(x, y, box_width, box_height)}")

    if not detections:
        print(
            "No match reached the minimum score of "
            f"{MINIMUM_MATCH_SCORE:.2f}. Try selecting a tighter, clearer "
            "dog-head template or lowering the score threshold."
        )

    cv2.imshow("Selected template", template)
    cv2.imshow("Multi-scale template matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
