import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from tensorflow.keras.applications import ResNet50
import os
from keras.models import load_model
# Preprocessing functions
def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def normalize_image(image):
    return image / 255.0

def augment_image(image):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
    angle = np.random.uniform(-10, 10)
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image

# Parsing functions
def parse_activity_labels(directory):
    activity_labels = {}
    video_id = os.path.basename(directory)
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            object_name = filename.split('.')[0]
            with open(os.path.join(directory, filename), 'r') as file:
                activity_labels[video_id] = []
                for line in file:
                    start_frame, end_frame, action_type = line.strip().split()
                    activity_labels[video_id].append({
                        'start_frame': int(start_frame),
                        'end_frame': int(end_frame),
                        'action_type': int(action_type),
                        'object_name': object_name
                    })
    return activity_labels

def parse_detection_tracking_annotations(file_path):
    annotations = {}
    tree = ET.parse(file_path)
    root = tree.getroot()
    video_id = os.path.splitext(os.path.basename(file_path))[0]
    annotations[video_id] = []
    for object in root.findall('object'):
        for polygon in object.findall('polygon'):
            frame = int(polygon.find('t').text)
            pts = polygon.findall('pt')
            x_coords = [int(pt.find('x').text) for pt in pts]
            y_coords = [int(pt.find('y').text) for pt in pts]
            annotations[video_id].append({
                'frame': frame,
                'x_coords': x_coords,
                'y_coords': y_coords
            })
    return annotations

def preprocess_video_frames(directory, size=(224, 224), augment=False):
    frames = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".jpg"):
                video_id = os.path.basename(root)
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = resize_image(image, size)
                image = normalize_image(image)
                if augment:
                    image = augment_image(image)
                if video_id not in frames:
                    frames[video_id] = []
                frames[video_id].append(image)
    return frames

def preprocess_and_yield_images(frames, activity_labels, video_ids):
    images = []
    labels = []
    for video_id in video_ids:
        for frame, label in zip(frames[video_id], activity_labels[video_id]):
            images.append(frame)
            labels.append(label)
    return np.array(images), np.array(labels)

def split_dataset(images, labels, test_size=0.2):
    if images is None or labels is None:
        print("Dataset is empty. Cannot split.")
        return None, None, None, None
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=test_size, random_state=42)
    return train_images, val_images, train_labels, val_labels

# Model creation and training functions
def create_activity_recognition_model(input_shape=(224, 224, 3), num_classes=5):
    # Load the ResNet50 model without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    return model

def train_activity_recognition_model(model, train_generator, val_generator, epochs=20, batch_size=32, early_stopping=True):
    if train_generator is None or val_generator is None:
        print("Training or validation dataset is empty. Cannot train.")
        return None

    # Early stopping callback
    if early_stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks = [early_stop]
    else:
        callbacks = []

    # Train the model
    history = model.fit(train_generator, epochs=epochs, batch_size=batch_size,
                        validation_data=val_generator, callbacks=callbacks)

    return history

def evaluate_model(model, val_generator):
    if val_generator is None:
        print("Validation dataset is empty. Cannot evaluate.")
        return None
    loss, accuracy = model.evaluate(val_generator, verbose=2)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image)
    image = normalize_image(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image_class(model, image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Main script
def main():
    # Preprocess the dataset
    activity_labels = parse_activity_labels('/Users/rohithr/Desktop/volvo_project/Data/Labels/104154')
    print("Activity labels parsed.")
    print(activity_labels.keys())

    # Parse detection and tracking annotations
    annotations = parse_detection_tracking_annotations('/Users/rohithr/Desktop/volvo_project/Data/DetectionTrackingAnnotations/104154.xml')
    print("Detection and tracking annotations parsed.")

    # Preprocess video frames
    frames = preprocess_video_frames('/Users/rohithr/Desktop/volvo_project/Data/frames_ce/104154', augment=True)
    print("Video frames preprocessed.")
    print(frames.keys())

    # Get the common video IDs
    video_ids = set(frames.keys()) & set(activity_labels.keys())
    print(video_ids)

    # Split into training and validation sets
    train_images, val_images, train_labels, val_labels = split_dataset(*preprocess_and_yield_images(frames, activity_labels, video_ids), test_size=0.2)

    # Create the model
    model = create_activity_recognition_model()

    # Create generators for traning and validation data
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_generator = train_datagen.flow(train_images, np.array([label['action_type'] for label in train_labels]), batch_size=32)
    val_generator = val_datagen.flow(val_images, np.array([label['action_type'] for label in val_labels]), batch_size=32)

    # Train the model
    history = train_activity_recognition_model(model, train_generator, val_generator, epochs=100, batch_size=64)
    if history is None:
        print("Failed to train the model. Exiting.")
        exit(1)

    # Evaluate the model
    evaluate_model(model, val_generator)
    #prediction
    model_path = '/Users/rohithr/Desktop/volvo_project/activity_recognition_model.h5'
    model = load_model(model_path)
    image_path = '/Users/rohithr/Desktop/volvo_project/dummy_dataset/simple_images/volvo construction bulldozer/volvo construction bulldozer_26.jpg'
    predicted_class = predict_image_class(model, image_path)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
