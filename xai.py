# %% [markdown]
# # Using LIME for explaining Image Classifiers
# ## CHAPTER 05 - *Practical Exposure of using LIME in ML*
# 
# From **Applied Machine Learning Explainability Techniques** by [**Aditya Bhattacharya**](https://www.linkedin.com/in/aditya-bhattacharya-b59155b6/), published by **Packt**

# %% [markdown]
# ### Objective
# 
# In this notebook, we will applying LIME to explain black-box image classifiers. Before starting with this notebook, I would recommend you to go through the concepts discussed in *Chapter 05 - Practical Exposure of using LIME in ML* and *Chapter 04 - Introduction to LIME for model interpretability* to have a better understanding on the code provided in the notebook.

# %% [markdown]
# ### Installing the modules

# %% [markdown]
# Install the following libraries in Google Colab or your local environment, if not already installed.

# %%
import pandas
import numpy 
import matplotlib 
# import seaborn 
import tensorflow 
import lime 
import skimage

# %% [markdown]
# ### Loading the modules

# %%
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c_map
from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import lime
from lime import lime_image
from lime import submodular_pick

from skimage.segmentation import mark_boundaries

np.random.seed(123)

# %%
print(f" Version of tensorflow used: {tf.__version__}")

# %% [markdown]
# ### Loading the data
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

load_model = load_model(os.path.join('model','ai_imageclassifier_100_epochs.h5'))
predictions = Dense(7, activation='softmax', name="dense_output")(load_model.output) 
model = Model(inputs=load_model.input, outputs=predictions)
# %% [markdown]
# Since we are more interested to check how black-box image classifiers can be explained using LIME, we will focus only on the inference part. Let us load any generic image data. For this example, we will take the data from this source: https://i.imgur.com/1Phh6hv.jpeg

# %%
# path = "./data/FAKE/"
# image_name = "___in_the_style_of..."
# dir = os.listdir(path)

def load_image_data_from_path(path, image_name):
    '''
    Function to load image data from directory
    '''
    # The local path to our target image
    image_path = os.path.join(path,image_name)
    display(Image(image_path))
    return image_path

IMG_SIZE = (256, 256)
def transform_image(image_path, size):
    '''
    Function to transform an image to normalized numpy array
    '''
    img = image.load_img(image_path, target_size=size)
    img = image.img_to_array(img)# Transforming the image to get the shape as [channel, height, width]
    img = np.expand_dims(img, axis=0) # Adding dimension to convert array into a batch of size (1,299,299,3)
    img = img/255.0 # normalizing the image to keep within the range of 0.0 to 1.0
    
    return img

path = "./data/REAL/"
dir_ = os.listdir(path)
save_path = "./xai/"
# index = 2
# print(f"We will deal with predicted class: {top5_pred[index][1]}")
# image_name = "abraham_storck_die_niederlandische_flotte_auf_der_reede_vor_amsterdam_0_in_the_style_of_baroque"
# image_path = load_image_data_from_path(path + ".jpg", image_name)

plt_num = 0

def explanation_heatmap(exp, exp_class):
    '''
    Using heat-map to highlight the importance of each super-pixel for the model prediction
    '''
    dict_heatmap = dict(exp.local_exp[exp_class])
    heatmap = np.vectorize(dict_heatmap.get)(exp.segments) 
    plt.figure(plt_num)
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    plt.savefig(save_path + image_name + "_heatmap.jpg")
    # plt.show()

for image_name in dir_:
    image_path = load_image_data_from_path(path, image_name)
    normalized_img = transform_image(image_path, IMG_SIZE)
    model_prediction = model.predict(normalized_img)
    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(normalized_img[0], 
                                 model.predict, 
                                 top_labels=5, 
                                 hide_color=0, 
                                 num_samples=1000)
    plt.figure(plt_num)
    plt.imshow(exp.segments)
    plt.axis('off')
    # plt.show()
    plt.savefig(save_path + image_name + "_exp.jpg")
    plt_num += 1
    explanation_heatmap(exp, exp.top_labels[0])
    plt_num += 1

# %% [markdown]
# Similar to other tutorials covered in this chapter where image classifiers are used. We will perform some initial data pre-processing with images.

# %%



# %% [markdown]
# ### Defining the model
# 
# For this example, we are not training a model from scratch, but rather defining a pretrained Tensorflow Xception model as our black-box Deep Learning model which we will be explaining using the LIME framework.

# %%


# def get_model_predictions(data):
#     model_prediction = model.predict(data)
#     print(f"The predicted class is : {decode_predictions(model_prediction, top=1)[0][0][1]}")
#     return decode_predictions(model_prediction, top=1)[0][0][1]

# plt.imshow(normalized_img[0])
# pred_orig = get_model_predictions(normalized_img)

# %% [markdown]
# The image is predicted as *Tiger Shark* which is the correct prediction and the black-box model is successfully able to give the correct prediction. Now, let us even take a look at the top 5 predictions along with the model confidences.

# %%
# model_prediction = model.predict(normalized_img)
# top5_pred = decode_predictions(model_prediction, top=5)[0]
# for pred in top5_pred:
#     print(pred[1])

# %% [markdown]
# As we see, although the model is well trained to produce the correct prediction, but there are chances that the model is not just looking into main object in the image but as well as the surrounding background. This is evident from the prediction of *scuba_driver* present in the top 5 prediction list. So, it is important for us understand, the key components or parts of the image the model is looking into to make the prediction.

# %% [markdown]
# ### Model Explanation with LIME

# %% [markdown]
# Now, we will use the LIME framework to identify "super-pixels" or image segments used by the model to predict the outcome.

# %%
# explainer = lime_image.LimeImageExplainer()

# %%
# exp = explainer.explain_instance(normalized_img[0], 
#                                  model.predict, 
#                                  top_labels=5, 
#                                  hide_color=0, 
#                                  num_samples=1000)

# %% [markdown]
# Our explainer object is ready, but let us visualize the various explanation segments created by the LIME algorithm.

# %%
# plt.imshow(exp.segments)
# plt.axis('off')
# plt.show()
# plt.savefig(image_name + ".jpg")

# %% [markdown]
# Now, let us use the top segments or super pixels to identify the region of interest of the image used by the model to make its prediction.

# %%
# def generate_prediction_sample(exp, exp_class, weight = 0.1, show_positive = True, hide_background = True):
#     '''
#     Method to display and highlight super-pixels used by the black-box model to make predictions
#     '''
#     image, mask = exp.get_image_and_mask(exp_class, 
#                                          positive_only=show_positive, 
#                                          num_features=6, 
#                                          hide_rest=hide_background,
#                                          min_weight=weight
#                                         )
#     plt.imshow(mark_boundaries(image, mask))
#     plt.axis('off')
#     plt.show()
    # plt.savefig("lime_gen_predict_sample" + str(show_positive) + str(hide_background) + "2.jpg")

# %%
# generate_prediction_sample(exp, exp.top_labels[0], show_positive = True, hide_background = True)

# # %% [markdown]
# # As we can see from the above image that the model was able to identify the correct region, which does indicate the correct prediction of the outcome by the model.

# # %%
# generate_prediction_sample(exp, exp.top_labels[0], show_positive = True, hide_background = False)

# # %%
# generate_prediction_sample(exp, exp.top_labels[0], show_positive = False, hide_background = False)

# %% [markdown]
# The above samples show us how we can hide or show the background along with the super-pixels or even outline or highlight the super-pixels to identify the region of interest used by the model to make the prediction. What we see from here does make sense, and does allow us to increase trust towards black-box models. We can also form a heat-map to show how important each super-pixel is to get some more granular explaianbility.

# %%

# %% [markdown]
# We can clearly identify the most influential segments used by the model to make the prediction using this heatmap visualization.

# %% [markdown]
# Now, let try to perform the same steps for another explanation class and see if the results are different.

# %%
# index = 2
# print(f"We will deal with predicted class: {top5_pred[index][1]}")

# # %%
# generate_prediction_sample(exp, exp.top_labels[index], weight = 0.0001, show_positive = False, hide_background = False)

# # %%
# explanation_heatmap(exp, exp.top_labels[index])

# %% [markdown]
# In this case, we are trying to find out what made the model predict the outcome as *hammerhead shark*. When we used the LIME explaianbility methods, the visualizations clearly show that the middle part of the shark along with its fin, does contribute positively towards predicting the outcome as *hammerhead shark*, but the face and the front part contribute negatively towards the prediction. This is quite consistent with our human knowledge as well. *Hammerhead Sharks* are also sharks, so the middle part and the fin looks similar to *Tiger Sharks* but the face or the front portion of the hammerhead sharks looks like the shape of a hammer, which is significantly different from that of a tiger shark.

# %% [markdown]
# ### Final Thoughts

# %% [markdown]
# As we clearly saw in this notebook, how LIME can be easily used to explain image classifiers. Next time, whenever you work on training Deep Learning models to classify images, I would strongly recommend you to try out LIME to explain your model and find out if the model is looking into right areas of the image to make the final prediction!

# %% [markdown]
# ## Reference

# %% [markdown]
# 1. The image is taken from Imgur: https://i.imgur.com/1Phh6hv.jpeg
# 2. LIME Open Source Python Framework in GitHub - https://github.com/marcotcr/lime
# 3. Research paper on LIME - [“Why Should I Trust You?”
# Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf)
# 4. Original blog post on LIME by the author - https://homes.cs.washington.edu/~marcotcr/blog/lime/
# 4. Some of the utility functions and code are taken from the GitHub Repository of the author - Aditya Bhattacharya https://github.com/adib0073

# %%



