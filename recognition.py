import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from memory_profiler import profile

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
    video_id = os.path.basename(directory) # Extract video ID from the subfolder name
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            object_name = filename.split('.')[0]
            with open(os.path.join(directory, filename), 'r') as file:
                activity_labels[video_id] = [] # Use the video ID as the key
                for line in file:
                    start_frame, end_frame, action_type = line.strip().split()
                    activity_labels[video_id].append({
                        'start_frame': int(start_frame),
                        'end_frame': int(end_frame),
                        'action_type': int(action_type),
                        'object_name': object_name # Include the object name for clarity
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

def split_dataset(images, labels, test_size=0.2):
    if images is None or labels is None:
        print("Dataset is empty. Cannot split.")
        return None, None, None, None
    return train_test_split(images, labels, test_size=test_size, random_state=42)

# Model creation and training functions
def create_activity_recognition_model(input_shape=(224, 224, 3), num_classes=5):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model

def train_activity_recognition_model(model, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32):
    if train_images is None or val_images is None:
        print("Training or validation dataset is empty. Cannot train.")
        return None
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(val_images, val_labels))
    return history

def evaluate_model(model, val_images, val_labels):
    if val_images is None:
        print("Validation dataset is empty. Cannot evaluate.")
        return None
    loss, accuracy = model.evaluate(val_images, val_labels, verbose=2)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Main script
file_log = open("memory_profiler2.log","w+")
@profile(stream = file_log)
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

    # Create lists to store images and labels
    video_ids = set(frames.keys()) & set(activity_labels.keys())
    print(video_ids)
    
    all_images = []
    all_labels = []
    for video_id in video_ids:
        for frame, label in zip(frames[video_id], [activity_labels[video_id]] * len(frames[video_id])):
            all_images.append(frame)
            all_labels.append(label)

    # Convert to NumPy arrays
    all_images = np.array(all_images)
    print(all_images.shape)
    all_labels = np.array(all_labels)
    print(all_labels.shape)

    # Check if dataset is not empty
    if len(all_images) > 0 and len(all_labels) > 0:
        # Split into training and validation sets
        train_images, val_images, train_labels, val_labels = split_dataset(all_images, all_labels)

        # Create the model
        model = create_activity_recognition_model()

        # Train the model
        history = train_activity_recognition_model(model, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32)
        if history is None:
            print("Failed to train the model. Exiting.")
            exit(1)

        # Evaluate the model
        evaluate_model(model, val_images, val_labels)
    else:
        print("Dataset is empty. Cannot proceed with training and evaluation.")

if __name__ == "__main__":
    main()
