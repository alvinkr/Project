import cv2
 
deploy_path = "C:/Users/alvin kurnia/Documents/data latihan/deploy.prototxt.txt"
model_path = "C:/Users/alvin kurnia/Documents/data latihan/res10_300x300_ssd_iter_140000.caffemodel"
face_cascade = cv2.CascadeClassifier('C:/Users/alvin kurnia/Documents/data latihan/haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromCaffe(deploy_path, model_path)
# Define the emotions
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Resize face ROI for emotion detection
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (48, 48), (0, 0, 0), swapRB=True, crop=False)

        # Feed face ROI into emotion detection model
        net.setInput(blob)
        output = net.forward()

        # Get predicted emotion
        emotion_index = output[0].argmax()
        # Print the output model
        #print("Output Model:", output[0])

        # Ensure emotion_index is within the valid range
        if 0 <= emotion_index < len(emotions):
            emotion = emotions[emotion_index]
        else:
            emotion = "no expression"

        # Draw rectangle around the face and display predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Expression Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()