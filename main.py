
import os
import cv2

faceCascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
eyeCascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"))
smileCascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml"))

font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)

while True:
    try:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Frame obtain / convert logic
        if cv2.waitKey(1) == 27:
            break

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.putText(frame, 'Face', (x, y), font, 0.75, (255, 0, 0), 2)

            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.16,
                minNeighbors=35,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sh, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                cv2.putText(frame, 'Mouth', (x + sx, y + sy), 1, 1, (0, 255, 0), 1)

            eyes = eyeCascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
                cv2.putText(frame, 'Eye', (x + ex, y + ey), 1, 1, (0, 255, 255), 1)

            cv2.putText(frame, 'Number of Faces : ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)
            # Display the resulting frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        break

