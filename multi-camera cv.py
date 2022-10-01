import sys
import cv2
import mediapipe as mp
import imutils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = mp.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = mp.zeros(image.shape, dtype=mp.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = mp.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:

####testing area

#--- WebCam1 (successful)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,950)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,920)

#--- WebCam2 (SUCCESSFUL! (error was caused because cap1 was not reffrenced or was getting misrefrenced by cap itself))

cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH,950)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,920)

cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 950)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,920)



####testing area

with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as pose:

  while cap.isOpened():
    success, image = cap.read()
    success1, image1 = cap.read()

    success, image = cap.read()
    success2, image2 = cap1.read(1)

    success, image = cap.read()
    success3, image3 = cap2.read(2)
    if not success:

      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)



    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Flip the image horizontally for a selfie-view display.

    cv2.imshow('MediaPipe Pose-1', cv2.flip(image, 0))
    if cv2.waitKey(5) & 0xFF == 27:
        break

    cv2.imshow('MediaPipe Pose-2', cv2.flip(image2, 0))
    if cv2.waitKey(5) & 0xFF == 27:
        break

    cv2.imshow('MediaPipe Pose-3', cv2.flip(image3, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()