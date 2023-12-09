#Se importa las librerias necesarias
import cv2
import numpy as np
import os
import pickle
from deepface import DeepFace

# Inicializar el reconocedor de rostros y el diccionario de nombres
if os.path.exists('face_recognizer.yml') and os.path.exists('name_dict.pkl'):
    # Si los archivos existen, cargar el modelo y el diccionario de nombres
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face_recognizer.yml')
    with open('name_dict.pkl', 'rb') as f:
        name_dict = pickle.load(f)

else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    name_dict = ['dummy1', 'dummy2']
    # Si los archivos no existen, crear un modelo y un diccionario de nombres dummy
    dummy_face1 = np.zeros((200, 200), dtype='uint8')
    dummy_face2 = np.ones((200, 200), dtype='uint8')
    labels = np.array([0, 1])
    faces = np.array([dummy_face1, dummy_face2])
    recognizer.train(faces, labels)

# Inicializar la captura de video
cap = cv2.VideoCapture(0)
# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Definir una lista de nombres de emociones
emotion_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

while True:
    # Capturar el frame de video
    ret, frame = cap.read()
    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraer el rostro del frame
        face = gray[y:y+h, x:x+w]
        # Predecir la etiqueta y la confianza con el reconocedor
        label, confidence = recognizer.predict(face)

        # Imprimir la etiqueta y la confianza
        print(f'Label = {name_dict[label]} with a confidence of {confidence}')
        tconfidence = 100 - confidence

        # Convertir el rostro a color
        color_face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)

        # Predecir la emoción del rostro
        result = DeepFace.analyze(color_face, actions=['emotion'], enforce_detection=False)
        # Imprimir el resultado
        #print(result)
        emotion = result[0]['dominant_emotion']

        # Si la confianza es mayor que 50, se reconoce la cara
        if confidence < 50:
            name = name_dict[label]
            # Dibujar un rectángulo alrededor de la cara
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Colocar el nombre y la confianza en el frame
            cv2.putText(frame, (name + ' ' + str(round(tconfidence,2)) + '%'), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f'Recognized face: {name}')
        else:
            # Si la confianza es menor que 50, la cara es desconocida
            name = 'Unknown'
            # Dibujar un rectángulo alrededor de la cara
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Colocar el nombre en el frame
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print('Unknown face detected.')
            # Si se presiona la tecla 'i', pedir un nombre y guardar el rostro
            if cv2.waitKey(1) & 0xFF == ord('i'):
                name = input('Enter name: ')
                # Agregar el nombre al diccionario de nombres
                name_dict.append(name)
                print('Saving face...' + name_dict[len(name_dict)-1])
                # Entrenar el reconocedor con los nuevos rostros y etiquetas
                recognizer.train(np.array([face]), np.array([len(name_dict)-1]))

        cv2.putText(frame, emotion, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Mostrar el frame
    cv2.imshow('Face Recognition', frame)

    # Si se presiona la tecla 'q', salir del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guardar el modelo y el diccionario de nombres
recognizer.write('face_recognizer.yml')
with open('name_dict.pkl', 'wb') as f:
    pickle.dump(name_dict, f)
print('Face saved.')

# Liberar la captura de video
cap.release()
# Cerrar todas las ventanas
cv2.destroyAllWindows()