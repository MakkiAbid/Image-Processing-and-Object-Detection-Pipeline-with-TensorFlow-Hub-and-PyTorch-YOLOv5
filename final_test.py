import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
import os
from shutil import copyfile
import torch
from matplotlib import pyplot as plt
import csv

# Load the model for blending 2 pictures
print("Loading the image stylization model...")
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("Model loaded successfully.")

def load_image(img_path):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def show_generated_image(img_path):
    generated_image = cv2.imread(img_path)
    generated_image_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
    plt.imshow(generated_image_rgb)
    plt.show()

def show_image_with_loss(img_path, content_loss):
    generated_image = cv2.imread(img_path)
    generated_image_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
    plt.imshow(generated_image_rgb)
    plt.title(f'Content Loss: {content_loss}')
    plt.show()

def process_images(content_dir, style_dir, output_dir):
    output_data = []
    success = True
    content_image_paths = [os.path.join(content_dir, img) for img in os.listdir(content_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    style_image_paths = [os.path.join(style_dir, img) for img in os.listdir(style_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

    for content_img_path in content_image_paths:
        content_img_base = os.path.splitext(os.path.basename(content_img_path))[0]
        
        for style_img_path in style_image_paths:
            style_img_base = os.path.splitext(os.path.basename(style_img_path))[0]

            output_subdir = os.path.join(output_dir, f'{content_img_base}_{style_img_base}')
            os.makedirs(output_subdir, exist_ok=True)

            content_image = load_image(content_img_path)
            if content_image is None:
                success = False
                continue

            style_image = load_image(style_img_path)
            if style_image is None:
                success = False
                continue

            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

            generated_img_name = f"{content_img_base}_{style_img_base}.jpg"
            generated_img_path = os.path.join(output_subdir, generated_img_name)

            cv2.imwrite(generated_img_path, cv2.cvtColor(np.squeeze(stylized_image) * 255, cv2.COLOR_BGR2RGB))

            input_img_paths = [content_img_path, style_img_path]
            for img_path in input_img_paths:
                img_name = os.path.basename(img_path)
                output_img_path = os.path.join(output_subdir, img_name)
                copyfile(img_path, output_img_path)

            # Load YOLOv5 model
            print("Loading YOLOv5 model for object detection...")
            yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            print("YOLOv5 model loaded successfully.")

            # Run object detection on the generated image
            print("Running object detection...")
            results = yolo_model(generated_img_path)

            # Create a directory for detected objects
            detected_objects_dir = os.path.join(output_subdir, 'Detected_Objects_YOLO5')
            os.makedirs(detected_objects_dir, exist_ok=True)

            # Crop and save detected objects
            if results.xyxy[0].shape[0] == 0:
                print("No object found in the image.")
                with open(os.path.join(detected_objects_dir, "no_object_found.txt"), "w") as f:
                    f.write("No object found in the image.")
            else:
                print("Saving detected objects...")
                for i, det in enumerate(results.xyxy[0]):
                    x_min, y_min, x_max, y_max, _, class_label = det.tolist()
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    original_img = cv2.imread(generated_img_path)
                    cropped_img = original_img[y_min:y_max, x_min:x_max]

                    # Get class name using appropriate method or attribute
                    class_name = yolo_model.names[int(class_label)]

                    # Change output name of cropped image to only the detected object name
                    cropped_img_name = f"{class_name}.jpg"
                    output_cropped_img_path = os.path.join(detected_objects_dir, cropped_img_name)
                    cv2.imwrite(output_cropped_img_path, cropped_img)

            # Calculate content loss using TensorFlow operations
            print("Calculating content loss...")
            vgg_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
            vgg_model.trainable = False

            content_image = load_image(content_img_path)
            generated_image = load_image(generated_img_path)

            def content_loss(content_features, stylized_features):
                return tf.reduce_mean(tf.square(content_features - stylized_features))

            # Preprocess images for VGG19
            content_image_preprocessed = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
            generated_image_preprocessed = tf.keras.applications.vgg19.preprocess_input(generated_image * 255)

            # Extract features from VGG19
            content_features = vgg_model(content_image_preprocessed)
            generated_features = vgg_model(generated_image_preprocessed)

            # Calculate content loss
            loss_value = content_loss(content_features, generated_features)

            # Show the generated image with content loss
            show_image_with_loss(generated_img_path, loss_value.numpy())
            print(f"Content Loss for {generated_img_path}: {loss_value.numpy()}")

            # Save output data
            output_data.append({'GENERATED_OUTPUT': generated_img_path, 'CONTENT_LOSS': loss_value.numpy()})

    # Save output data to CSV
    csv_file_path = os.path.join(os.getcwd(), 'output_data.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['GENERATED_OUTPUT', 'CONTENT_LOSS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)
    print("Results saved to output_data.csv.")

# Define directories
content_images_dir = './test_images'
style_images_dir = './style_images'
output_dir = './Output_Images'

# Process images
process_images(content_images_dir, style_images_dir, output_dir)
