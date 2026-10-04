"""Demonstrate directional line counting and counting in a rectangular area."""

import cv2
import imutils
import numpy as np
from centroid_tracker import CentroidTracker


def create_tracker():
    return CentroidTracker(max_disappeared=40, max_distance=50)


def detect_centroids(frame, background_subtractor):
    foreground_mask = background_subtractor.apply(frame)
    _, foreground_mask = cv2.threshold(foreground_mask, 250, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_OPEN, kernel, iterations=2
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_DILATE, kernel, iterations=2
    )
    contours, _ = cv2.findContours(
        foreground_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    input_centroids = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        input_centroids.append((int(x + width / 2.0), int(y + height / 2.0)))
    return input_centroids, foreground_mask


def count_line_crossing(track, centroid, line_y):
    if track["counted_line"]:
        return 0, 0
    previous_positions = [point[1] for point in track["centroids"]]
    direction = centroid[1] - np.mean(previous_positions)
    if centroid[1] < line_y and direction < 0:
        track["counted_line"] = True
        return 1, 0
    if centroid[1] > line_y and direction > 0:
        track["counted_line"] = True
        return 0, 1
    return 0, 0


def count_area_entry(track, centroid, area_points):
    if track["counted_area"]:
        return 0
    if cv2.pointPolygonTest(area_points, (centroid[0], centroid[1]), False) < 0:
        return 0
    track["counted_area"] = True
    return 1


def count_tracked_people(objects, trackable_objects, line_y, area_points):
    new_in = 0
    new_out = 0
    new_area = 0
    for object_id, centroid in objects.items():
        track = trackable_objects.get(object_id)
        if track is None:
            track = {"centroids": [centroid], "counted_line": False, "counted_area": False}
        else:
            track["centroids"].append(centroid)
            entered, exited = count_line_crossing(track, centroid, line_y)
            new_in += entered
            new_out += exited
            new_area += count_area_entry(track, centroid, area_points)
        trackable_objects[object_id] = track
    return new_in, new_out, new_area


def draw_tracking(frame, objects, line_coordinates, area_points, counts):
    for object_id, centroid in objects.items():
        cv2.putText(
            frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    cv2.line(frame, line_coordinates[0], line_coordinates[1], (255, 0, 0), 2)
    cv2.polylines(frame, [area_points], isClosed=True, color=(0, 0, 255), thickness=2)
    labels = ("Count In", "Count Out", "Area Count")
    colors = ((0, 255, 255), (0, 255, 255), (255, 0, 255))
    for index, (label, count, color) in enumerate(zip(labels, counts, colors)):
        cv2.putText(
            frame, f"{label}: {count}", (10, 50 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )


def main(video_path="people3.mp4"):
    counts = np.zeros(3, dtype=int)
    line_coordinates = [(100, 200), (500, 200)]
    area_points = np.array([[300, 150], [500, 150], [500, 350], [300, 350]])
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    capture = cv2.VideoCapture(video_path)
    tracker = create_tracker()
    trackable_objects = {}
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = imutils.resize(frame, width=600)
            input_centroids, foreground_mask = detect_centroids(frame, background_subtractor)
            objects = tracker.update(input_centroids)
            counts += count_tracked_people(
                objects, trackable_objects, line_coordinates[0][1], area_points
            )
            draw_tracking(frame, objects, line_coordinates, area_points, counts)
            cv2.imshow("Frame", frame)
            cv2.imshow("FG Mask", foreground_mask)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
