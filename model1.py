# Import tensorflow to use it backend
import tensorflow as tf
#Import ImageDataGenarator to do image processes easily
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import os

import matplotlib.pyplot as plt
import numpy as np

#BATCH SIZE means how many files will feed the data in one training tour
BATCH_SIZE=50
#Our images are not on same size but our model has one input size
#So we will make all the images same size for training
IMG_SHAPE=150
total_train=9013
total_val=1002

train_dir=os.path.join('base/train')
validation_dir=os.path.join('base/validation')

# Normally one pixel value is between 0-255 but we will make between 0-1 by rescaling the image
train_image_generator      = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data


train_data_flow = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                            directory=train_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_SHAPE,IMG_SHAPE),  #(150,150)
                                                            class_mode="categorical",
                                                            )

validation_data_flow = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                      directory=validation_dir,
                                                                      shuffle=False,
                                                                      target_size=(IMG_SHAPE,IMG_SHAPE),  #(150,150)
                                                                      class_mode="categorical"
                                                                      )
# Create the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# How many round all the data will train
EPOCHS = 10
history = model.fit_generator(
    train_data_flow,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=validation_data_flow,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)


# Take the values from training history to show the
# training processes value accuracy and loos changes
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()