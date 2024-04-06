# %%
import tensorflow as tf
import os

# %%
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import json
import shap

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)

# %%
import cv2
from os import listdir
from matplotlib import pyplot as plt
import numpy as np

# %%
import imghdr
from PIL import Image 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

path = "./data/FAKE/"
dir_ = os.listdir(path)

for image in dir_:
    file = os.path.join(path,image)
    if not imghdr.what(file):
        print(file)
        #if you want to remove them, uncomment next line
        os.remove(file) 
        continue
    f_img = path+"/"+image
    img = Image.open(f_img)
    if (img.size != (256,256)):
        img = img.resize((256,256))
        print(f_img)
        img.save(f_img)

folder_dir = "./data/FAKE"
for image in os.listdir(folder_dir):
    f_img = folder_dir+"/"+image
    img = Image.open(f_img)
    new_f_img = f_img.replace(".png", ".jpg")
    os.remove(f_img)
    img.save(new_f_img)

path = "./data/REAL/"
dir_ = os.listdir(path)

for image in dir_:
    file = os.path.join(path,image)
    if not imghdr.what(file):
        print(file)
        os.remove(file) 
        continue
    f_img = path+"/"+image
    img = Image.open(f_img)
    if (img.size != (256,256)):
        img = img.resize((256,256))
        print(f_img)
        img.save(f_img)

folder_dir = "./data/REAL"
for image in os.listdir(folder_dir):
    f_img = folder_dir+"/"+image
    img = Image.open(f_img)
    new_f_img = f_img.replace(".png", ".jpg")
    os.remove(f_img)
    img.save(new_f_img)

img = cv2.imread(os.path.join('realism', 'FAKE', 'a_y_jackson_the_winter_road_quebec_1921_in_the_style_of_art_nouveau.png'))

# %%
# img.shape

# %% [markdown]
### Loading the data

# %%
data = tf.keras.utils.image_dataset_from_directory('data', image_size=(256, 256))

# %%
data_it = data.as_numpy_iterator()

# %%
batch = data_it.next()

# %%
fig, ax = plt.subplots(ncols=8, figsize=(10, 10))
for idx, img in enumerate(batch[0][:8]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# %% [markdown]
# ### Preprocessing

# %%
print(batch[0].min())
print(batch[0].max())

# %% [markdown]
# Scaling the data between 0 and 1

# %%
data = data.map(lambda x, y : (x/255, y))

# %%
scaled_it = data.as_numpy_iterator()

# %%
batch = scaled_it.next()

# %%
print(batch[0].min())
print(batch[0].max())

# %% [markdown]
# Splitting the data into training, cross-validation and testing sets

# %%
print("len(data)", len(data))

# %%
train_size = int(len(data)*0.01)
cv_size = int(len(data)*0.2)
test_size = int(len(data)*0.2) #+1?

# %%
print("total_size", train_size+cv_size+test_size)

# %%
train = data.take(train_size)
cv = data.skip(train_size).take(cv_size)
test = data.skip(train_size+cv_size).take(test_size)

# %%
# print("total len", len(train)+len(cv)+len(test))
print("training data on", len(train) * 30, "images, len(train)")
print("training data on", train_size * 30, "images, train_size")

folder = "data/"

import os
images = sorted(os.listdir(folder)) #["frame_00", "frame_01", "frame_02", ...]

from PIL import Image 
import numpy as np 

img_array = []
for image in images:
    im = Image.open(folder + image)
    img_array.append(np.asarray(im)) #.transpose(1, 0, 2))

img_array = np.array(img_array)
print(img_array.shape)
# (75, 50, 100, 3)

# %% [markdown]
## Building the DL Model

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# %%
model = Sequential()

# # %%
model.add(Conv2D(16, (4, 4), 1, activation='relu', input_shape=(256, 256, 3))) #(256, 256, 3)
model.add(MaxPooling2D())

model.add(Conv2D(32, (4, 4), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (4, 4), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# # %%
model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# %%
# model.summary()

# %%
logdir = 'logs'

# %%
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# # %%
hist = model.fit(train, epochs=100, validation_data=cv, callbacks=[tensorboard_callback])
# %% [markdown]
# ### Performance

# %%
fig = plt.figure()
plt.plot(hist.history['loss'], color='blue', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='cv_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
plt.savefig("output-loss-100-renaissance-dalle.jpg")

# %%
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='blue', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='cv_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
plt.savefig("output-accuracy-100-renaissance-dalle.jpg")

model = load_model(os.path.join('model','ai_imageclassifier_100_epochs_AN_SD.h5'))
# %%
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# # %%
pre = Precision()
rec = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    rec.update_state(y, yhat)
    acc.update_state(y, yhat)

# # %%
print(f'Precision: {pre.result().numpy()}, Recall: {rec.result().numpy()}, Accuracy: {acc.result().numpy()}')
# %%
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# %%
model = load_model(os.path.join('model','ai_imageclassifier_100_epochs.h5'))

# %%
img = cv2.imread('./data/FAKE/mary_cassatt_maternity_in_the_style_of_impressionism.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
# plt.savefig("test_impressionism_sd.jpg")

# %%
resize = tf.image.resize(img, (256, 256))

# %%
y_pred = model.predict(np.expand_dims(resize/255, 0))

# %%
y_pred

# %%
if y_pred > 0.5: 
    print(f'Predicted class: REAL')
else:
    print(f'Predicted class: AI')
