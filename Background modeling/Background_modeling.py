import cv2
import numpy as np

# 1. INITIALIZATION & BACKGROUND MODELING
# Logic adapted from example3.py to calculate Mean/Median background

video_source = (r'J:\My Drive\StudiaDokumenty\Mgr\2semestr\Dozór wizyjny\Laboratorium\Lab1 - Modelowanie tła\highway.mp4')
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print('Unable to open file')
    exit(0)

print("Calculating background model (Mean and Median)... please wait.")

frameIds = []
# Collect frame IDs like in Example 3
count = 0
while count < 101:
    frameIds.append(count)
    count += 1

frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

# Calculate the mean along the time axis
meanFrame = np.mean(frames, axis=0).astype(dtype=np.uint8)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

print("Background models calculated.")

# Select which model to use (Task requires Average, but option for Median)
# Change this variable to 'median' to use the median filtration
model_type = 'mean' 

if model_type == 'mean':
    backgroundModel = meanFrame
    print("Using MEAN background model.")
else:
    backgroundModel = medianFrame
    print("Using MEDIAN background model.")

# Convert background model to grayscale for thresholding
backgroundModel_gray = cv2.cvtColor(backgroundModel, cv2.COLOR_BGR2GRAY)


# 2. MAIN VIDEO PROCESSING LOOP
# Logic adapted from example5.py for the loop and example4.py for subtraction

# Reset video to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, currentFrame = cap.read()
    
    if ret is True:
        # Convert current frame to grayscale
        currentFrame_gray = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between current frame and background 
        difference = cv2.absdiff(currentFrame_gray, backgroundModel_gray)
        
        # Apply threshold to create the binary foreground mask 
        # Threshold value (25) can be adjusted based on lighting/noise
        _, fgMask = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
        
        # Display results
        cv2.imshow('Original Frame', currentFrame)
        cv2.imshow('Foreground Mask (Subtraction)', fgMask)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    else:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()