import cv2

cap = cv2.VideoCapture("silent_videos/track_0.mid.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
