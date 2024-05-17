import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam


train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)
# Path to your dataset-

train_data_dir = r'A:\Programs\pythonProject\pythonProject\emotion\data\images\train'
validation_data_dir = r'A:\Programs\pythonProject\pythonProject\emotion\data\images\test'
img_width, img_height = 48, 48  # Image dimensions
input_shape = (img_width, img_height)
# Loading training data
train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    color_mode='grayscale',  # For grayscale images
    class_mode='categorical')

# Loading validation data
validation_generator = validation_data_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    color_mode='grayscale',  # For grayscale images
    class_mode='categorical')



# Create new model
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))


model1.add(Flatten())
model1.add(Dense(1024,  activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(7,  activation='softmax'))
 #  Assuming 7 emotions to detect



# Compile model
model1.compile(loss='categorical_crossentropy',
               optimizer=Adam(learning_rate=0.0001),
               metrics=['accuracy'])

# Train the model
history = model1.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64)

model_json = model1.to_json()
with open(r'A:\Programs\pythonProject\pythonProject\emotion\data\model\model2.json', 'w') as json_file:
    json_file.write(model_json)

# Save the trained model
model1.save_weights(r'A:\Programs\pythonProject\pythonProject\emotion\data\model\model2.h5')
