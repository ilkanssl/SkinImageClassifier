import os

# Create a new directory for the images
base = 'base'
os.mkdir(base)

# Create a directory for the train files and validation files
train = os.path.join(base, 'train')
validation = os.path.join(base, 'validation')
os.mkdir(train)
os.mkdir(validation)

# Create new file pathes and names for the folders
akiec = os.path.join(train, 'akiec')
bcc = os.path.join(train, 'bcc')
bkl = os.path.join(train, 'bkl')
df = os.path.join(train, 'df')
mel = os.path.join(train, 'mel')
nv = os.path.join(train, 'nv')
vasc = os.path.join(train, 'vasc')

#Creating the folders
os.mkdir(akiec)
os.mkdir(bcc)
os.mkdir(bkl)
os.mkdir(df)
os.mkdir(mel)
os.mkdir(nv)
os.mkdir(vasc)

# Create new file pathes and names for the folders
akiec_v = os.path.join(validation, 'akiec')
bcc_v = os.path.join(validation, 'bcc')
bkl_v = os.path.join(validation, 'bkl')
df_v = os.path.join(validation, 'df')
mel_v = os.path.join(validation, 'mel')
nv_v = os.path.join(validation, 'nv')
vasc_v = os.path.join(validation, 'vasc')

#Creating the folders
os.mkdir(akiec_v)
os.mkdir(bcc_v)
os.mkdir(bkl_v)
os.mkdir(df_v)
os.mkdir(mel_v)
os.mkdir(nv_v)
os.mkdir(vasc_v)



