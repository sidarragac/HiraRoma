import os
import shutil
import random
import zipfile
from glob import glob
import gc

import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "hiragana_dataset/hiragana"
OUTPUT_PATH = "hiragana_split"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

AUGMENTATIONS_PER_IMAGE = 5
ALPHAS_TO_TEST = [0.0001, 0.001, 0.01]


# ============================================================
# DATA PREPARATION
# ============================================================

def extract_dataset():
    with zipfile.ZipFile("hiragana.zip", 'r') as zip_ref:
        zip_ref.extractall("hiragana_dataset")


def create_folder_structure(base_path, classes):
    for split in ["train", "val", "test"]:
        split_path = os.path.join(base_path, split)
        os.makedirs(split_path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(split_path, cls), exist_ok=True)


def split_data(file_list):
    random.shuffle(file_list)
    n = len(file_list)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    return file_list[:train_end], file_list[train_end:val_end], file_list[val_end:]


def get_augmenter():
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.03, scale_limit=0.05, rotate_limit=10,
            border_mode=0, p=0.7
        ),
        A.GridDistortion(
            num_steps=3, distort_limit=0.2, p=0.3
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    ])


def augment_and_save(image_path, save_path, augmentations, augmenter):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: unable to read {image_path}")
        return
    
    cv2.imwrite(save_path, image)

    for i in range(augmentations):
        augmented = augmenter(image=image)['image']
        base, ext = os.path.splitext(os.path.basename(image_path))
        new_filename = f"{base}_aug{i}{ext}"
        cv2.imwrite(os.path.join(os.path.dirname(save_path), new_filename), augmented)


def prepare_dataset():
    extract_dataset()
    
    random.seed(42)
    augmenter = get_augmenter()
    
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    print(f"Classes found: {classes}")
    
    create_folder_structure(OUTPUT_PATH, classes)
    
    for cls in classes:
        class_path = os.path.join(DATASET_PATH, cls)
        files = glob(os.path.join(class_path, "*.*"))
        train_files, val_files, test_files = split_data(files)
        
        for f in val_files:
            shutil.copy(f, os.path.join(OUTPUT_PATH, "val", cls))
        for f in test_files:
            shutil.copy(f, os.path.join(OUTPUT_PATH, "test", cls))
        
        for f in train_files:
            save_path = os.path.join(OUTPUT_PATH, "train", cls, os.path.basename(f))
            augment_and_save(f, save_path, AUGMENTATIONS_PER_IMAGE, augmenter)
    
    print("Data split and augmentation complete!")


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

def count_images_per_class(split_path):
    classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    counts = {}
    for cls in classes:
        files = glob(os.path.join(split_path, cls, "*.*"))
        counts[cls] = len(files)
    return counts


def print_stats(split_name, counts):
    values = list(counts.values())
    print(f"--- {split_name.upper()} SET ---")
    print(f"Total classes: {len(counts)}")
    print(f"Total images: {sum(values)}")
    print(f"Mean images per class: {np.mean(values):.2f}")
    print(f"Median images per class: {np.median(values)}")
    print(f"Min images per class: {np.min(values)}")
    print(f"Max images per class: {np.max(values)}")
    print()


def plot_distributions(all_counts):
    plt.figure(figsize=(18, 5))
    for i, (split_name, counts) in enumerate(all_counts.items(), 1):
        classes = list(counts.keys())
        values = list(counts.values())
        plt.subplot(1, 3, i)
        sns.barplot(x=classes, y=values)
        plt.title(f"{split_name} set class distribution")
        plt.xticks(rotation=90)
        plt.ylabel("Image count")
        plt.xlabel("Class")
    plt.tight_layout()
    plt.show()


def show_sample_images(split_path, classes, samples_per_class=3):
    plt.figure(figsize=(15, len(classes)*3))
    idx = 1
    for cls in classes:
        files = glob(os.path.join(split_path, cls, "*.*"))
        for f in files[:samples_per_class]:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(len(classes), samples_per_class, idx)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
            idx += 1
    plt.tight_layout()
    plt.show()


def perform_eda():
    splits = ["train", "val", "test"]
    all_counts = {}
    
    for split in splits:
        path = os.path.join(OUTPUT_PATH, split)
        counts = count_images_per_class(path)
        print_stats(split, counts)
        all_counts[split] = counts
    
    plot_distributions(all_counts)
    
    train_classes = list(all_counts["train"].keys())
    sample_classes = random.sample(train_classes, min(5, len(train_classes)))
    print(f"Showing samples from classes: {sample_classes}")
    show_sample_images(os.path.join(OUTPUT_PATH, "train"), sample_classes)


# ============================================================
# DATA GENERATORS
# ============================================================

def albumentations_augmenter(img):
    augmenter = get_augmenter()
    img = np.uint8(img)
    augmented = augmenter(image=img)
    return augmented["image"]


def create_data_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: preprocess_input(albumentations_augmenter(x))
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_gen = train_datagen.flow_from_directory(
        "hiragana_split/train",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        "hiragana_split/val",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )
    
    test_gen = test_datagen.flow_from_directory(
        "hiragana_split/test",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen


def save_label_encoder(train_gen):
    class_names = list(train_gen.class_indices.keys())
    le = LabelEncoder()
    le.fit(class_names)
    joblib.dump(le, "label_encoder.pkl")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def create_feature_extractor():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model


def extract_features_and_save(generator, save_path_features, save_path_labels, model):
    features = []
    labels = []
    
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        feats_batch = model.predict(x_batch, verbose=0)
        feats_batch = feats_batch.reshape(feats_batch.shape[0], -1)
        features.append(feats_batch)
        labels.append(y_batch)
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    np.save(save_path_features, features)
    np.save(save_path_labels, labels)
    
    print(f"Guardado: {save_path_features}, {save_path_labels}")


def extract_all_features(train_gen, val_gen, test_gen):
    model = create_feature_extractor()
    
    extract_features_and_save(train_gen, "X_train_feat.npy", "y_train_enc.npy", model)
    extract_features_and_save(val_gen, "X_val_feat.npy", "y_val_enc.npy", model)
    extract_features_and_save(test_gen, "X_test_feat.npy", "y_test_enc.npy", model)


# ============================================================
# MODEL TRAINING
# ============================================================

def train_svm_with_validation():
    print("Cargando features de entrenamiento...")
    X_train_feat = np.load("X_train_feat.npy")
    y_train_enc = np.load("y_train_enc.npy")
    
    X_val_feat = np.load("X_val_feat.npy")
    y_val_enc = np.load("y_val_enc.npy")
    
    print("Entrenando SVM...")
    best_acc = 0
    best_alpha = None
    best_model = None
    
    for alpha in ALPHAS_TO_TEST:
        print(f"\nProbando modelo con alpha={alpha}")
        svm_clf = SGDClassifier(loss="hinge", alpha=alpha, max_iter=1000, tol=1e-3)
        svm_clf.fit(X_train_feat, y_train_enc)
        val_pred = svm_clf.predict(X_val_feat)
        val_acc = accuracy_score(y_val_enc, val_pred)
        print(f"Validación Accuracy = {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_alpha = alpha
            best_model = svm_clf
    
    print(f"\nMejor modelo encontrado: alpha={best_alpha} con accuracy={best_acc:.4f}")
    
    joblib.dump(best_model, "svm_hiragana_model.pkl")
    
    del X_train_feat, y_train_enc
    gc.collect()
    
    return best_model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(svm_clf):
    print("Cargando features de test...")
    X_test_feat = np.load("X_test_feat.npy")
    y_test_enc = np.load("y_test_enc.npy")
    
    print("Prediciendo y evaluando...")
    y_pred = svm_clf.predict(X_test_feat)
    
    print("Accuracy:", accuracy_score(y_test_enc, y_pred))
    print(classification_report(y_test_enc, y_pred))
    
    return y_test_enc, y_pred


def plot_confusion_matrix(y_test_enc, y_pred, le):
    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='weighted')
    recall = recall_score(y_test_enc, y_pred, average='weighted')
    
    print("\n===== MÉTRICAS FINALES =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (ponderado): {f1:.4f}")
    print(f"Recall (ponderado): {recall:.4f}")
    
    cm = confusion_matrix(y_test_enc, y_pred)
    labels = le.classes_
    
    plt.figure(figsize=(30, 30))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True,
        linewidths=0.3,
        linecolor='white'
    )
    
    plt.title("Matriz de Confusión - Hiragana SVM", fontsize=16, pad=20)
    plt.xlabel("Predicciones", fontsize=14, labelpad=10)
    plt.ylabel("Etiquetas Verdaderas", fontsize=14, labelpad=10)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=10)
    for text in plt.gca().texts:
        text.set_fontsize(7)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# INFERENCE
# ============================================================

def convert_to_jpg(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer {image_path}")
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(output_path, img)


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.stack([img]*3, axis=-1)
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)


def predict_image(image_path):
    with zipfile.ZipFile("mios.zip", 'r') as zip_ref:
        zip_ref.extractall("mios")
    
    model = create_feature_extractor()
    svm_clf = joblib.load("svm_hiragana_model.pkl")
    le = joblib.load("label_encoder.pkl")
    
    ruta_imagen = "mios/mis_hiraganas/ma"
    convert_to_jpg(ruta_imagen + ".png", ruta_imagen + ".jpg")
    image = preprocess_image(ruta_imagen + ".jpg")
    image = preprocess_image("image.png")
    
    features = model.predict(image)
    features = features.reshape(1, -1).astype(np.float64)
    
    pred = svm_clf.predict(features)
    pred = pred.astype(int)
    pred_label = le.inverse_transform(pred)[0]
    
    print("Predicción:", pred_label)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    prepare_dataset()
    perform_eda()
    
    train_gen, val_gen, test_gen = create_data_generators()
    save_label_encoder(train_gen)
    
    extract_all_features(train_gen, val_gen, test_gen)
    
    svm_clf = train_svm_with_validation()
    y_test_enc, y_pred = evaluate_model(svm_clf)
    
    le = joblib.load("label_encoder.pkl")
    plot_confusion_matrix(y_test_enc, y_pred, le)