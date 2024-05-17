from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from keras.models import model_from_json

#face_classifier = cv2.CascadeClassifier(r'A:\Programs\pythonProject\pythonProject\emotion\data\haarcascades\haarcascade_frontalface_default.xml')
#classifier = load_model(r'A:\Programs\pythonProject\pythonProject\emotion\emotion.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open(r'A:\Programs\pythonProject\pythonProject\emotion\data\model\emotion_model.json')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)

model1.load_weights(r'A:\Programs\pythonProject\pythonProject\emotion\data\model\model1.h5')
print('loaded model from disk')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier(r'A:\Programs\pythonProject\pythonProject\emotion\data\haarcascades\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = model1.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
