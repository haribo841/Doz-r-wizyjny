import cv2
import numpy as np
import imutils
from collections import OrderedDict

# --- TRACKER CLASS ---
class CentroidTracker:
    def __init__(self, maxDisappeared=40, maxDistance=70):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids):
        if len(inputCentroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - np.array(inputCentroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                if D[row, col] > self.maxDistance: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

# --- CONFIGURATION ---
EXCLUDE_OVERLAP_THRESHOLD = 0.4 
SHOW_DEBUG_SKIPPED = True 

cap = cv2.VideoCapture('people3.mp4')
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Video not found or cannot be opened.")

first_frame = imutils.resize(first_frame, width=600)
roi = cv2.selectROI('Select billboard ROI', first_frame, False)
cv2.destroyWindow('Select billboard ROI')
bx, by, bw, bh = tuple(map(int, roi))

# High sensitivity MOG2
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)
ct = CentroidTracker(maxDisappeared=50, maxDistance=80)
trackableObjects = {}
line_y = 250
count_people = 0

# Kernels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = imutils.resize(frame, width=600)
    height, width = frame.shape[:2]

    # --- STEP 1: LIGHT PRE-PROCESSING ---
    fgMask = backSub.apply(frame, learningRate=-1)
    fgMask[fgMask == 127] = 0  # Remove shadows
    # Median Blur 3 is much less destructive for distant people than 5
    fgMask = cv2.medianBlur(fgMask, 3) 
    _, fgMask = cv2.threshold(fgMask, 10, 255, cv2.THRESH_BINARY)

    # --- STEP 2: SOFT MORPHOLOGY ---
    # As requested: OPEN(1), CLOSE(1), DILATE(1)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)

    # Hard-clear billboard ROI
    fgMask[by:by+bh, bx:bx+bw] = 0

    # --- STEP 3: SENSITIVE WATERSHED ---
    dist_transform = cv2.distanceTransform(fgMask, cv2.DIST_L2, 5)
    # Lowered threshold to 0.1 to catch local peaks in smaller/flatter blobs
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(cv2.dilate(fgMask, kernel, iterations=2), sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers[by:by+bh, bx:bx+bw] = 1 

    markers = cv2.watershed(cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR), markers)
    separated_mask = np.zeros_like(fgMask)
    separated_mask[markers > 1] = 255

    # --- STEP 4: DETECTION ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(separated_mask, connectivity=8)
    inputCentroids = []

    for lbl in range(1, num_labels):
        x_l, y_l, w_l, h_l, area_l = stats[lbl]
        
        # Lowered area threshold to 80 for distant figures
        if area_l < 80: continue 

        # Check billboard overlap
        overlap_pixels = 0
        x1, y1, x2, y2 = max(bx, x_l), max(by, y_l), min(bx + bw, x_l + w_l), min(by + bh, y_l + h_l)
        if x1 < x2 and y1 < y2:
            overlap_pixels = cv2.countNonZero((labels[y1:y2, x1:x2] == lbl).astype(np.uint8))
        
        if (overlap_pixels / float(area_l)) >= EXCLUDE_OVERLAP_THRESHOLD:
            if SHOW_DEBUG_SKIPPED:
                cv2.rectangle(frame, (x_l, y_l), (x_l + w_l, y_l + h_l), (255, 0, 255), 1)
            continue

        # Logic for splitting groups
        if (w_l / float(h_l)) > 1.3:
            inputCentroids.append((int(x_l + w_l/4), int(y_l + h_l/2)))
            inputCentroids.append((int(x_l + 3*w_l/4), int(y_l + h_l/2)))
        else:
            inputCentroids.append((int(centroids[lbl][0]), int(centroids[lbl][1])))
        
        cv2.rectangle(frame, (x_l, y_l), (x_l + w_l, y_l + h_l), (0, 255, 0), 1)

    # --- STEP 5: TRACKING & COUNTING ---
    objects = ct.update(inputCentroids)
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, {"centroids": [], "counted": False})
        to["centroids"].append(centroid)
        
        if not to["counted"]:
            # Increased vertical window slightly to ensure high-speed walkers are caught
            if (line_y - 25) < centroid[1] < (line_y + 25):
                count_people += 1
                to["counted"] = True
        
        trackableObjects[objectID] = to
        cv2.putText(frame, f"{objectID}", (centroid[0]-5, centroid[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # --- DISPLAY ---
    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
    cv2.putText(frame, f'Count: {count_people}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow('Kamera', frame)
    cv2.imshow('Maska (Debug)', separated_mask)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()