# Install TensorFlow and Keras
!pip install tensorflow==2.12.0
!pip install keras==2.12.0

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Bird Image/Bird Images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    '/content/drive/MyDrive/Bird Image/Bird Images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define the model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Replace num_classes with the number of bird species
])

# ... (Previous code from your notebook remains the same) ...
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Replace num_classes with the number of bird species
])
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,  # Adjust the number of epochs as needed
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Save the trained model to Google Drive
model_save_path = '/content/drive/MyDrive/bird_classifier.h5'  # Update with your desired path
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for image display

# Load the trained model from Google Drive
model_path = '/content/drive/MyDrive/bird_classifier.h5'  # Path where you saved the model
model = keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.  # Rescale pixel values
    return img

# Path to the image you want to classify
image_path = '/content/drive/MyDrive/Bird Image/Bird Images/007.Parakeet_Auklet/Parakeet_Auklet_0001_795972.jpg'

# Preprocess the image
processed_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(processed_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Assuming you have a list of bird species names:
class_names = [
    'Black_footed_Albatross', 'Laysan_Albatros','Sooty_Albatros','Groove_billed_Ani','Crested_Auklet','Least_Auklet',
    'Parakeet_Auklet','Rhinoceros_Auklet','Brewer_BlackBird','Red_winged_Blackbird']  # Replace with your class names

# Print the predicted class
predicted_class = class_names[predicted_class_index]
print("Predicted class:", predicted_class)


# Display the image
img = image.load_img(image_path, target_size=(224, 224))  # Load image for display
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')  # Turn off axis labels
plt.show()
