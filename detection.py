import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Taking my webcam...
# Put any video files
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()

    # This will set the image to grayscaled_image
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # This will Print out the image.
    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)

    # press the letter 'q' to quit.
    if key==81 or key==113:
        break

webcam.release()

print("Program Done")