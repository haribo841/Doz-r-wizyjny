"""Count people near a line after excluding a manually selected billboard."""

import cv2
import imutils
import numpy as np
from centroid_tracker import CentroidTracker

EXCLUDE_OVERLAP_THRESHOLD = 0.4


def create_tracker():
    return CentroidTracker(max_disappeared=50, max_distance=80)


def create_separated_mask(frame, background_subtractor, roi, kernel, kernel_close):
    billboard_x, billboard_y, billboard_width, billboard_height = roi
    foreground_mask = background_subtractor.apply(frame, learningRate=-1)
    foreground_mask[foreground_mask == 127] = 0
    foreground_mask = cv2.medianBlur(foreground_mask, 3)
    _, foreground_mask = cv2.threshold(foreground_mask, 10, 255, cv2.THRESH_BINARY)
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1
    )
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)
    billboard_rows = slice(billboard_y, billboard_y + billboard_height)
    billboard_columns = slice(billboard_x, billboard_x + billboard_width)
    foreground_mask[billboard_rows, billboard_columns] = 0

    distance_transform = cv2.distanceTransform(foreground_mask, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(
        distance_transform, 0.1 * distance_transform.max(), 255, 0
    )
    sure_foreground = np.uint8(sure_foreground)
    unknown = cv2.subtract(
        cv2.dilate(foreground_mask, kernel, iterations=2), sure_foreground
    )
    _, markers = cv2.connectedComponents(sure_foreground)
    markers += 1
    markers[unknown == 255] = 0
    markers[billboard_rows, billboard_columns] = 1
    markers = cv2.watershed(
        cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR), markers
    )
    separated_mask = np.zeros_like(foreground_mask)
    separated_mask[markers > 1] = 255
    return separated_mask


def detect_centroids(frame, separated_mask, roi, show_debug_skipped):
    billboard_x, billboard_y, billboard_width, billboard_height = roi
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        separated_mask, connectivity=8
    )
    input_centroids = []
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area < 80:
            continue
        overlap_pixels = 0
        overlap_left = max(billboard_x, x)
        overlap_top = max(billboard_y, y)
        overlap_right = min(billboard_x + billboard_width, x + width)
        overlap_bottom = min(billboard_y + billboard_height, y + height)
        if overlap_left < overlap_right and overlap_top < overlap_bottom:
            overlap_pixels = cv2.countNonZero(
                (
                    labels[overlap_top:overlap_bottom, overlap_left:overlap_right]
                    == label
                ).astype(np.uint8)
            )
        if overlap_pixels / float(area) >= EXCLUDE_OVERLAP_THRESHOLD:
            if show_debug_skipped:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 255), 1)
            continue
        if width / float(height) > 1.3:
            input_centroids.append((int(x + width / 4), int(y + height / 2)))
            input_centroids.append((int(x + 3 * width / 4), int(y + height / 2)))
        else:
            input_centroids.append((int(centroids[label][0]), int(centroids[label][1])))
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)
    return input_centroids


def count_tracked_people(objects, trackable_objects, line_y):
    newly_counted = 0
    for object_id, centroid in objects.items():
        track = trackable_objects.get(object_id, {"centroids": [], "counted": False})
        track["centroids"].append(centroid)
        if not track["counted"] and line_y - 25 < centroid[1] < line_y + 25:
            newly_counted += 1
            track["counted"] = True
        trackable_objects[object_id] = track
    return newly_counted


def draw_tracking(frame, objects, line_y, count_people):
    for object_id, centroid in objects.items():
        cv2.putText(
            frame, f"{object_id}", (centroid[0] - 5, centroid[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
        )
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
    cv2.putText(
        frame, f"Count: {count_people}", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )


def main(video_path="people3.mp4", show_debug_skipped=True):
    capture = cv2.VideoCapture(video_path)
    try:
        success, first_frame = capture.read()
        if not success:
            raise RuntimeError("Video not found or cannot be opened.")
        first_frame = imutils.resize(first_frame, width=600)
        roi = tuple(map(int, cv2.selectROI("Select billboard ROI", first_frame, False)))
        cv2.destroyWindow("Select billboard ROI")
        background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=30, detectShadows=True
        )
        tracker = create_tracker()
        trackable_objects = {}
        line_y = 250
        count_people = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = imutils.resize(frame, width=600)
            separated_mask = create_separated_mask(
                frame, background_subtractor, roi, kernel, kernel_close
            )
            input_centroids = detect_centroids(
                frame, separated_mask, roi, show_debug_skipped
            )
            objects = tracker.update(input_centroids)
            count_people += count_tracked_people(objects, trackable_objects, line_y)
            draw_tracking(frame, objects, line_y, count_people)
            cv2.imshow("Kamera", frame)
            cv2.imshow("Maska (Debug)", separated_mask)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
