import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class EPIDetector:
    def __init__(self, dataset_dir='dataset', img_size=128, batch_size=16, epochs=10):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.labels = []

    def preprocess_data(self):
        datagen = ImageDataGenerator(rescale=1./255)

        self.train_data = datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        self.val_data = datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'val'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.test_data = datagen.flow_from_directory(
            os.path.join(self.dataset_dir, 'test'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.labels = list(self.train_data.class_indices.keys())


    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(self.labels), activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        print("[INFO] Iniciando o treinamento...")
        self.model.fit(self.train_data, validation_data=self.val_data, epochs=self.epochs)
        print("[INFO] Treinamento finalizado.")
    
    def evaluate(self):
        if self.test_data:
            loss, acc = self.model.evaluate(self.test_data)
            print(f"[INFO] Acurácia no conjunto de teste: {acc:.2f}")

    def save_model(self, model_path='modelo_epi.h5'):
        self.model.save(model_path)
        print(f"[INFO] Modelo salvo em '{model_path}'.")

    def load_model(self, model_path='modelo_epi.h5'):
        self.model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Modelo carregado de '{model_path}'.")

    def predict_frame(self, frame):
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img_array = np.expand_dims(img / 255.0, axis=0)
        predictions = self.model.predict(img_array)
        return self.labels[np.argmax(predictions)]

    def run_realtime_detection(self):
        print("[INFO] Iniciando detecção em tempo real. Pressione 'q' para sair.")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            label = self.predict_frame(frame)
            cv2.putText(frame, f'Detectado: {label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Detecção EPI", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
