
# Import pandas to read .csv file
import pandas as pd

# Import os for file operations
import os

# Import train_test_split to split images for training and testing easily
from sklearn.model_selection import train_test_split

# Import shutil to copy files from one directory to another
import shutil


# Reads the file and returns a data frame
df = pd.read_csv('HAM10000_metadata.csv')

# Returns dx values(labels) column
class_labels = df['dx']

# Split data into training and validation datas
# 'test_size=0.1' means %10 of data will be test data and %90 will be training data
# 'stratify=class_labels' means data will split refer to class labels values
train_df, validation_df = train_test_split(df, test_size=0.1, stratify=class_labels)


# Set the index
df.set_index('image_id', inplace=True)

# Get a list of images in each of the two folders
folder_1 = os.listdir('ham10000_images_part_1')
folder_2 = os.listdir('ham10000_images_part_2')

# Get a list of train and val images
train_list = list(train_df['image_id'])
validation_list = list(validation_df['image_id'])

# Write file paths to use them easily
base = 'base'
train = os.path.join(base, 'train')
validation = os.path.join(base, 'validation')


# Move training images to corresponding folder
for image in train_list:

    imageId = image + '.jpg'
    label = df.loc[image, 'dx']

    if imageId in folder_1:
        # source path to image
        src = os.path.join('ham10000_images_part_1', imageId)
        # destination path to image
        dst = os.path.join(train, label, imageId)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if imageId in folder_2:
        # source path to image
        src = os.path.join('ham10000_images_part_2', imageId)
        # destination path to image
        dst = os.path.join(train, label, imageId)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

# Move validation images to corresponding folder
for image in validation_list:

    imageId = image + '.jpg'
    label = df.loc[image, 'dx']

    if imageId in folder_1:
        # source path to image
        src = os.path.join('ham10000_images_part_1', imageId)
        # destination path to image
        dst = os.path.join(validation, label, imageId)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if imageId in folder_2:
        # source path to image
        src = os.path.join('ham10000_images_part_2', imageId)
        # destination path to image
        dst = os.path.join(validation, label, imageId)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)