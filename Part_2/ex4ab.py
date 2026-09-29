#!/usr/bin/env python3

import cv2
from pathlib import Path


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
    _, score, _, location = cv2.minMaxLoc(response)
    return location, (template_width, template_height), score


def show_match(scene, template, title, minimum_score=0.5):
    location, size, score = locate_template(scene, template)
    width, height = size
    result = scene.copy()
    if score >= minimum_score:
        cv2.rectangle(
            result,
            location,
            (location[0] + width, location[1] + height),
            (0, 0, 255),
            2,
        )
        label = f"match score: {score:.3f}"
    else:
        label = f"No reliable match (best score: {score:.3f})"

    cv2.putText(
        result,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    print(f"{title}: location={location}, score={score:.3f}")
    cv2.imshow(title, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def main():
    images_directory = Path(__file__).resolve().parent / "images"
    template = load_image(images_directory / "modelo.png")
    scene = load_image(images_directory / "cenario.jpg")

    show_match(scene, template, "4a - cenario.jpg")

    height, width = scene.shape[:2]
    reduced_scene = cv2.resize(
        scene, (round(width * 0.6), round(height * 0.6))
    )
    template_height, template_width = template.shape[:2]
    reduced_template = cv2.resize(
        template,
        (round(template_width * 0.6), round(template_height * 0.6)),
        interpolation=cv2.INTER_AREA,
    )
    show_match(
        reduced_scene,
        reduced_template,
        "4b - cenario.jpg and template at 60%",
    )

    snow_scene = load_image(images_directory / "cenario_2.jpg")
    show_match(snow_scene, template, "4b - cenario_2.jpg")


if __name__ == "__main__":
    main()
