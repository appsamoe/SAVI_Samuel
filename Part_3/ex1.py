#!/usr/bin/env python3

import cv2
import numpy as np


# ==================== EXERCISE 3: IDENTIFY VEHICLE COLOR ====================
def identify_vehicle_color(hsv_frame, box):
    """Estimate body color from the central, upper part of a vehicle box."""
    x, y, box_width, box_height = box

    # Sample the car body rather than the dark shadow below its bounding box.
    left = x + int(box_width * 0.3)
    right = x + int(box_width * 0.7)
    if box_height < 180:
        top = y + int(box_height * 0.05)
        bottom = y + int(box_height * 0.4)
    else:
        top = y + int(box_height * 0.35)
        bottom = y + int(box_height * 0.65)
    hsv_crop = hsv_frame[top:bottom, left:right]
    pixels = hsv_crop.reshape(-1, 3)
    if len(pixels) == 0:
        return "unknown"

    # Bright, low-saturation paint is usually white or silver.
    bright_paint = (pixels[:, 1] < 50) & (pixels[:, 2] >= 130)
    if np.mean(bright_paint) >= 0.25:
        return "white/silver"

    # Median HSV values reduce the effect of glare, windows, and dark shadows.
    hue, saturation, brightness = np.median(pixels, axis=0)
    if saturation >= 55:
        if hue <= 10 or hue >= 160:
            return "red"
        if hue <= 24:
            return "brown"
        if hue <= 34:
            return "yellow"
        if hue <= 84:
            return "green"
        if hue <= 129:
            return "blue"
        return "purple"
    if saturation >= 40 and brightness >= 50:
        if 35 <= hue <= 84:
            return "green"
        if 85 <= hue <= 129:
            return "blue"

    # Low-saturation or very dark paint is classified by brightness.
    if brightness >= 190:
        return "white"
    if brightness >= 90:
        return "gray/silver"
    return "black"


def main():
    video_path = "docs/traffic.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MOG2 learns the fixed scene over time; create it once, outside the loop.
    background = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    # These thresholds keep small road markings out while retaining vehicle blobs.
    kernel = np.ones((3, 3), np.uint8)
    min_area = 5000
    max_area = int(width * height * 0.08)
    min_box_width, min_box_height = 20, 16
    min_aspect_ratio, max_aspect_ratio = 0.45, 1.8

    # Limit detection to the carriageway, without cutting off the left lane.
    road_polygon = np.array(
        [
            [0.07 * width, 0.99 * height],
            [0.88 * width, 0.99 * height],
            [0.66 * width, 0.04 * height],
            [0.41 * width, 0.04 * height],
        ],
        dtype=np.int32,
    )
    road_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(road_mask, [road_polygon], 255)

    # Tracks can start in the middle of the frame and remain briefly unmatched.
    start_y = int(height * 0.35)
    end_y = int(height * 0.92)
    max_missed = 45
    hits_to_confirm = 5

    # A vehicle is counted once when its tracked center crosses this line.
    count_line_y = int(height * 0.72)
    duplicate_window = 15
    duplicate_overlap = 0.25

    # ==================== EXERCISE 2: COUNT VEHICLES PER LANE ====================
    # At the counting line, these x-positions divide the four lanes from left to
    # right. Adjust them if the line or camera framing changes.
    lane_names = ["Left", "Middle-left", "Middle-right", "Right"]
    lane_boundaries = [
        int(width * 0.34),
        int(width * 0.515),
        int(width * 0.69),
    ]
    lane_counts = {name: 0 for name in lane_names}

    # Exercise 3 records the color for each vehicle that Exercise 1 counts.
    vehicle_color_reports = []

    tracks = {}
    next_id = 1
    count_events = []
    vehicle_count = 0
    frame_number = 0

    try:
        while True:
            read_ok, frame = cap.read()
            if not read_ok:
                break
            frame_number += 1

            # Keep only recent crossings for the duplicate-track check.
            count_events = [
                event
                for event in count_events
                if frame_number - event[0] <= duplicate_window
            ]

            # Keep definite foreground pixels, then remove specks and fill gaps.
            foreground = background.apply(frame)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            binary = cv2.threshold(
                foreground, 254, 255, cv2.THRESH_BINARY
            )[1]
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            clean_mask = cv2.bitwise_and(binary, road_mask)

            # Turn each connected foreground region into a filtered car candidate.
            component_count, _, stats, centers = (
                cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
            )
            detections = []
            for component in range(1, component_count):
                x, y, box_width, box_height, area = map(
                    int, stats[component]
                )
                aspect_ratio = box_width / box_height
                if (
                    area < min_area
                    or area > max_area
                    or box_width < min_box_width
                    or box_height < min_box_height
                    or not min_aspect_ratio <= aspect_ratio <= max_aspect_ratio
                ):
                    continue

                detections.append(
                    {
                        "center": centers[component],
                        "box": (x, y, box_width, box_height),
                    }
                )

            # Match each existing track to the nearest plausible detection.
            possible_matches = []
            for track_id, track in tracks.items():
                predicted_center = (
                    track["center"] + track["velocity"] * (track["missed"] + 1)
                )
                for detection_index, detection in enumerate(detections):
                    box_scale = max(
                        *track["box"][2:],
                        *detection["box"][2:],
                    )
                    max_distance = min(
                        140.0,
                        max(50.0, box_scale * 1.5) * (track["missed"] + 1),
                    )
                    distance = float(
                        np.linalg.norm(predicted_center - detection["center"])
                    )
                    if distance <= max_distance:
                        possible_matches.append(
                            (distance, track_id, detection_index)
                        )

            # Greedy nearest-first matching prevents reusing a track or detection.
            possible_matches.sort()
            matched_tracks = set()
            matched_detections = set()
            for _, track_id, detection_index in possible_matches:
                if (
                    track_id in matched_tracks
                    or detection_index in matched_detections
                ):
                    continue

                track = tracks[track_id]
                detection = detections[detection_index]
                previous_center = track["center"]

                # Remember when a vehicle moves downward across the count line.
                if (
                    previous_center[1] < count_line_y
                    and detection["center"][1] >= count_line_y
                ):
                    track["crossed_line"] = True
                    # Estimate the x-position exactly where the track crosses.
                    crossing_fraction = (
                        (count_line_y - previous_center[1])
                        / (detection["center"][1] - previous_center[1])
                    )
                    track["crossing_x"] = (
                        previous_center[0]
                        + crossing_fraction
                        * (detection["center"][0] - previous_center[0])
                    )
                    # Exercise 3: estimate this car's color at the crossing.
                    track["color"] = identify_vehicle_color(
                        hsv_frame, detection["box"]
                    )

                # Smooth velocity so the next frame can be matched through gaps.
                measured_velocity = (
                    detection["center"] - previous_center
                ) / (track["missed"] + 1)
                track["velocity"] = (
                    0.5 * track["velocity"] + 0.5 * measured_velocity
                )
                track["center"] = detection["center"]
                track["box"] = detection["box"]
                track["missed"] = 0
                track["hits"] += 1
                matched_tracks.add(track_id)
                matched_detections.add(detection_index)

            # Retain tracks through short occlusions; discard stale ones.
            for track_id in list(tracks):
                if track_id not in matched_tracks:
                    tracks[track_id]["missed"] += 1
                    tracks[track_id]["hits"] = 0
                    if tracks[track_id]["missed"] > max_missed:
                        del tracks[track_id]

            # Start tracks only in the useful part of the road.
            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detections:
                    continue
                if not start_y <= detection["center"][1] <= end_y:
                    continue

                tracks[next_id] = {
                    "center": detection["center"],
                    "velocity": np.zeros(2, dtype=np.float64),
                    "box": detection["box"],
                    "missed": 0,
                    "hits": 1,
                    "crossed_line": False,
                    "crossing_x": None,
                    "counted": False,
                    "duplicate": False,
                }
                next_id += 1

            # Count confirmed line crossings, ignoring overlapping duplicate tracks.
            for track_id, track in tracks.items():
                if (
                    track["counted"]
                    or track["duplicate"]
                    or track["hits"] < hits_to_confirm
                    or not track["crossed_line"]
                ):
                    continue

                x, y, box_width, box_height = track["box"]
                box_area = box_width * box_height
                is_duplicate = False
                for _, event_box in count_events:
                    old_x, old_y, old_width, old_height = event_box
                    overlap_width = max(
                        0, min(x + box_width, old_x + old_width) - max(x, old_x)
                    )
                    overlap_height = max(
                        0, min(y + box_height, old_y + old_height) - max(y, old_y)
                    )
                    overlap = overlap_width * overlap_height
                    if overlap / min(
                        box_area, old_width * old_height
                    ) >= duplicate_overlap:
                        is_duplicate = True
                        break

                if is_duplicate:
                    track["duplicate"] = True
                else:
                    track["counted"] = True
                    count_events.append((frame_number, track["box"]))
                    vehicle_count += 1
                    crossing_x = track["crossing_x"]
                    lane_index = sum(
                        crossing_x >= boundary for boundary in lane_boundaries
                    )
                    lane_name = lane_names[lane_index]
                    car_color = track.get("color", "unknown")
                    lane_counts[lane_name] += 1
                    # Keep one color-and-lane report for each counted vehicle.
                    vehicle_color_reports.append(
                        (track_id, lane_name, car_color)
                    )
                    print(
                        f"Vehicle {track_id} counted: "
                        f"lane = {lane_name}, color = {car_color}; "
                        f"total count = {vehicle_count}"
                    )

            # Draw tracked boxes, labels, the count line, and the running total.
            display = frame.copy()
            for track_id, track in tracks.items():
                x, y, box_width, box_height = track["box"]
                if track["duplicate"]:
                    color, label = (0, 165, 255), f"duplicate {track_id}"
                elif track["counted"]:
                    color, label = (0, 255, 0), f"ID {track_id}"
                else:
                    color, label = (255, 200, 0), f"new {track_id}"

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + box_width, y + box_height),
                    color,
                    2,
                )
                cv2.putText(
                    display,
                    label,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.line(
                display,
                (int(width * 0.18), count_line_y),
                (int(width * 0.88), count_line_y),
                (255, 0, 255),
                2,
            )
            # Mark where the count line separates the four lanes.
            for boundary in lane_boundaries:
                cv2.line(
                    display,
                    (boundary, count_line_y - 10),
                    (boundary, count_line_y + 10),
                    (0, 255, 255),
                    2,
                )
            cv2.putText(
                display,
                f"Count: {vehicle_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Vehicle tracking", display)
            # cv2.imshow("Clean foreground mask", clean_mask)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                print("Stopped by user.")
                break
    finally:
        # Release video and close windows even if processing is interrupted.
        cap.release()
        cv2.destroyAllWindows()

    print("Total vehicles detected =", vehicle_count)

    # ==================== EXERCISE 2 SUMMARY: VEHICLES PER LANE ====================
    print("\nVehicles per lane:")
    for lane_name in lane_names:
        print(f"{lane_name}: {lane_counts[lane_name]}")

    # ==================== EXERCISE 3 SUMMARY: COLOR OF EACH VEHICLE ====================
    print("\nVehicle colors:")
    for vehicle_number, (track_id, lane_name, color) in enumerate(
        vehicle_color_reports, start=1
    ):
        print(
            f"Vehicle {vehicle_number} (track {track_id}, {lane_name} lane): "
            f"{color}"
        )


if __name__ == "__main__":
    main()
