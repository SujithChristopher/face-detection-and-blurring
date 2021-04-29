import cv2

cap = cv2.VideoCapture(0)
while True:

    ret, image = cap.read()

    face_detect = cv2.CascadeClassifier(r'scr\haarcascade_frontalface_alt.xml')
    face_data = face_detect.detectMultiScale(image, 1.1, 5)

    # Draw rectangle around the faces which is our region of interest (ROI)
    for (x, y, w, h) in face_data:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y + h, x:x + w]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        image[y:y + roi.shape[0], x:x + roi.shape[1]] = roi

    # Display the output
    cv2.imshow("img", image)

    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
