"""Shared nearest-centroid tracking for the two people-counting demos."""

from collections import OrderedDict

import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=70):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def _mark_disappeared(self, object_ids):
        expired_ids = []
        for object_id in object_ids:
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                expired_ids.append(object_id)
        # Delete only after the dictionary traversal is complete.
        for object_id in expired_ids:
            self.deregister(object_id)

    def _match_centroids(self, distances, object_ids, input_centroids):
        rows = distances.min(axis=1).argsort()
        columns = distances.argmin(axis=1)[rows]
        used_rows = set()
        used_columns = set()
        for row, column in zip(rows, columns):
            if row in used_rows or column in used_columns:
                continue
            if distances[row, column] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[column]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_columns.add(column)
        return used_rows, used_columns

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            self._mark_disappeared(self.disappeared)
            return self.objects
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        object_ids = tuple(self.objects)
        object_centroids = tuple(self.objects.values())
        distances = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
            axis=2,
        )
        used_rows, used_columns = self._match_centroids(
            distances, object_ids, input_centroids
        )
        if distances.shape[0] >= distances.shape[1]:
            unused_rows = set(range(distances.shape[0])).difference(used_rows)
            self._mark_disappeared(object_ids[row] for row in unused_rows)
        else:
            unused_columns = set(range(distances.shape[1])).difference(used_columns)
            for column in unused_columns:
                self.register(input_centroids[column])
        return self.objects
