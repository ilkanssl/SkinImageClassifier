import os

# Counts of training images for each folder
print("Training image counts for each folder with alphabetical order:")
print(len(os.listdir('base/train/akiec')))
print(len(os.listdir('base/train/bcc')))
print(len(os.listdir('base/train/bkl')))
print(len(os.listdir('base/train/df')))
print(len(os.listdir('base/train/mel')))
print(len(os.listdir('base/train/nv')))
print(len(os.listdir('base/train/vasc')))

# Counts of validation images for each folder
print("Validation image counts for each folder with alphabetical order:")
print(len(os.listdir('base/validation/akiec')))
print(len(os.listdir('base/validation/bcc')))
print(len(os.listdir('base/validation/bkl')))
print(len(os.listdir('base/validation/df')))
print(len(os.listdir('base/validation/mel')))
print(len(os.listdir('base/validation/nv')))
print(len(os.listdir('base/validation/vasc')))


