import cv2
import os 
import numpy as np

dataPath = 'C:/Users/oriol/TdR2/Data'
peoplelist= os.listdir(dataPath)
print('Llista de persones: ', peoplelist)

labels = []
facesData = []
label = 0 #enumerador de totes les persones guardades a 'Data'

for nameDir in peoplelist: 
    personPath = dataPath + '/' + nameDir
    print('Llegint imatges')
    
    for fileName in os.listdir(personPath):
        print('Rostres: ', nameDir + '/' + fileName)
        labels.append(label) #labels es la variable contadora de persones. 
        
        #coloquem les imatges en escala de grisos
        facesData.append(cv2.imread(personPath + '/' + fileName, 0)), 
        image = cv2.imread(personPath + '/' + fileName, 0)
        
        #per a realitzar prova: veure com el sistema entra a cada carpeta i llegir les imatges de la base de dates
        
        ###################################
        #cv2.imshow('image', image)
        #cv2.waitKey(10) #el temps
        ###################################

    label = label + 1
        
#cv2.destroyAllWindows()

###########################################
#print('labels = ', labels)
#print('Nombre etiquetes 0: ', np.count_nonzero(np.array(labels)==0))
#print('Nombre etiquetes 0: ', np.count_nonzero(np.array(labels)==1))

#METODE O MODEL PER A ENTRENAR EL RECONEIXEMENT. EN AQUEST CAS, DELS TRES POSIBLES MODELS UTILITZO LUNIC QUE EM FUNCIONA

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()


print('Entrenant la xarxa...')

face_recognizer.train(facesData, np.array(labels)) #labels son les etiquetes que hem creat al principi

#per a no haver de repetir tot el proces dentrenament cada cop que safegeix una persona a la base de dates
#guardare aquest model que creare com a .xml per a que estigui sempre llest per a insertarse a la xarxa

face_recognizer.write('ModelFaceFrontal2.xml')
#face_recognizer.read('ModelFaceFrontal.xml')
print ('Model guardat.')