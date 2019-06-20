# Import libraires
import os
import shutil
import numpy as np

# Import ImageDataGenerator to generate augmented image
from keras.preprocessing.image import ImageDataGenerator


# We won't augmented the 'nv' because it already has over 6000 images
classes = {'akiec', 'bcc', 'bkl', 'df', 'mel', 'vasc'}

for item in classes:

    # Create a temporary directory for the augmented images
    tempDir = 'tempDir'
    os.mkdir(tempDir)

    # Create a directory within the base dir to store images of the same class
    img_dir = os.path.join(tempDir, 'img_dir')
    os.mkdir(img_dir)

    # Choose a class
    img_class = item

    # List all the images in the directory
    img_list = os.listdir('base/train/' + img_class)

    # Copy images from the class train dir to the img_dir
    for image_id in img_list:
        # source path to image
        src = os.path.join('base/train/' + img_class, image_id)
        # destination path to image
        dst = os.path.join(img_dir, image_id)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    # point to a dir containing the images and not to the images themselves
    path = tempDir
    save_path = 'base/train/' + img_class

    # Create a data generator to augment the images in real time
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        # brightness_range=(0.9,1.1),
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='jpg',
                                              target_size=(224, 224),
                                              batch_size=batch_size)

    # Generate the augmented images and add them to the training folders
    num_aug_images_wanted = 6000  # total number of images we want to have in each class
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

    # run the generator and create about 6000 augmented images
    for i in range(0, num_batches):
        imgs, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree('tempDir')
