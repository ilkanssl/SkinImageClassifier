import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


BATCH_SIZE=175
test_dir=os.path.join('base/test2')
test_image_generator = ImageDataGenerator(rescale=1./255)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


test_data_flow = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                      directory=test_dir,
                                                                      shuffle=False,
                                                                      target_size=(224,224),  #(224,224)
                                                                      class_mode="categorical"
                                                                      )
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

test_images,test_labels=next(test_data_flow)
for a in test_labels:
    print(a)
#plots(test_images,titles=test_labels)


plt.figure(figsize=(50,50))
for i in range(BATCH_SIZE):
    plt.subplot(7,25,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(test_labels[i])])
plt.show()


def create_model():
    model= tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
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

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy'])
    return model


# Create a basic model instance
model = create_model()
model.summary()

checkpoint_path = "model2save.ckpt"
model.load_weights(checkpoint_path)

loss,acc=model.evaluate_generator(test_data_flow)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

predictions=model.predict_generator(test_data_flow,verbose=1)

y_pred = np.rint(predictions)
y_true = test_data_flow.classes

print(y_true)


for i,pred in enumerate(predictions):
    print(np.argmax(pred))
    #print(np.argmax(test_labels[i]))
    print("------------------")


plt.figure(figsize=(50,50))
for i in range(BATCH_SIZE):
    plt.subplot(7,BATCH_SIZE/7,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(predictions[i])])
plt.show()



