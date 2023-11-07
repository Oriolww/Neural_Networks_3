import cv2 
import os 


dataPath = 'C:/Users/oriol/TdR2/Data'
imgPaths = os.listdir(dataPath)

print('imgPaths = ', imgPaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read('ModelFaceFrontal2.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True: 
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces: 
        rostre = auxFrame[y: y+h, x:x+w]
        rostre = cv2.resize(rostre, (150, 150), interpolation = cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostre)

        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

#si ho posem massa alt (5000, sempre donara un valor positiu, mai et dira que no coneix el rostre. 
# Si el baixem massa, tindrà dificultats reconeixent-lo, hem de trobar el terme mig.)

        if result[1] < 85:  # Si el valor de la quadrícula(dalt a l'esquerra) es més petit que el valor introudit, 
                             #reconeixerà a la cara (una altra cosa es que la identifiqui). Si es més
                             #gran, no el reconeixerà,

            cv2.putText(frame, '{}'.format(imgPaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        else: 
            cv2.putText(frame, 'No identificat', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)

    if k == 27: #esc
        break  

cap.release()
cv2.destroyAllWindows()