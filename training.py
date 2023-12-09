import cv2,os
import numpy as np
from PIL import Image
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create();
path="training"
name_dict=[]

def getImagesWithID(path):
    global name_dict
    id = 0
    folderPath = [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for person in folderPath:
        imagePaths=[os.path.join(person,f) for f in os.listdir(person)]
        for imagePath in imagePaths:
            faceImg=Image.open(imagePath).convert('L')
            faceNp=np.array(faceImg,'uint8')
            faces.append(faceNp)
            IDs.append(id)
            name_dict.append(os.path.basename(person.split('\\')[1]))
            cv2.imshow("training",faceNp)
            cv2.waitKey(10)
            id += 1
    return np.array(IDs), faces

Ids, faces = getImagesWithID(path)
recognizer.train(faces, Ids)
recognizer.save('face_recognizer.yml')
with open('name_dict.pkl', 'wb') as f:
    pickle.dump(name_dict, f)



