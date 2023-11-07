import cv2
import os 
import imutils

#crear els directoris on s'emmagatzemaran les imatges de manera automatitzada

personName = 'Guillem_Eras'
dataPath = 'C:/Users/oriol/TdR2/Data'
personPath = dataPath + '/' + personName

########################################################

if not os.path.exists(personPath):
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath) #indicar on es creara la nova carpeta


#programar la captura de video a la webcam del dispositiu.

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


count = 0


while True:
    
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=320) #frame es la variable que grada la capturacio del vid. Redimensionem la imatge a 320
    
 
#convertir totes les imatges en escala de grisos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy() #fem una copia de cad frame
    
    faces = faceClassif.detectMultiScale(gray, 1.3 )
    
#crear el rectange que identificara a la persona

    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cara = auxFrame[y:y + h, x:x + w]
        cara = cv2.resize(cara, (720, 720), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/cara_{}.jpg'. format(count), cara) #nom que rebra cada una de les fotos
        count = count + 1
        
    cv2.imshow('frame', frame) #el nom que li donem a la captura de video
    
#per a que la capturacio de video pari

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break
     
    
cap.release()
cv2.destroyAllWindows()
    