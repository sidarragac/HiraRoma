import os
import cv2
import numpy as np
import joblib

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

class ImagePredictor:
    def __init__(self):
        self.model = self.create_feature_extractor() # VGG16 Feature Extractor
        self.svm_clf = joblib.load("./utils/ml_model/svm_hiragana_model.pkl") # Support Vector Machine trained model
        self.le = joblib.load("./utils/ml_model/label_encoder.pkl") # Label Encoder for decoding predictions

    def create_feature_extractor(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model = Model(inputs=base_model.input, outputs=base_model.output)
        return model

    def convert_to_jpg(self, image_path, output_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"No se pudo leer {image_path}")
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(output_path, img)

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = np.stack([img]*3, axis=-1)
        img = preprocess_input(img.astype(np.float32))
        return np.expand_dims(img, axis=0)

    def predict_image(self, image_path):
        if image_path.lower().endswith('.png'):
            new_image_path = image_path.split('.')[0] + '.jpg'
            self.convert_to_jpg(image_path, new_image_path)
            os.remove(image_path)
            image_path = new_image_path
        image = self.preprocess_image(image_path)
        
        features = self.model.predict(image)
        features = features.reshape(1, -1).astype(np.float64)
        
        pred = self.svm_clf.predict(features)
        pred = pred.astype(int)
        pred_label = self.le.inverse_transform(pred)[0]
        
        print("Predicción:", pred_label)
        os.remove(image_path)

        return str(pred_label)

    def predict_images_in_folder(self, folder_path):
        image_files = os.listdir(folder_path)

        predictions = []

        for image_file in image_files:
            path = os.path.join(folder_path, image_file)
            pred_label = self.predict_image(path)
            predictions.append(pred_label)

        return predictions