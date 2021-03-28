import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add

model = load_model('./model_weights/model_9.h5')
# model._make_predict_function()  # 


model_img = ResNet50(weights="imagenet",input_shape=(224,224,3))
# model_img.summary()

model_new = Model(model_img.input,model_img.layers[-2].output)
# model_new._make_predict_function()

def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((1,feature_vector.shape[1]))
    return feature_vector


max_len=35

with open("./Storage/word_2_idx.pkl","rb") as f:
    word_2_idx = pickle.load(f)

with open("./Storage/idx_2_word.pkl","rb") as f:
    idx_2_word = pickle.load(f)


def predict_caption(photo):
    
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_2_idx[w] for w in in_text.split() if w in word_2_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_2_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def caption_this_image(image):
    enc = encode_image(image)
    caption = predict_caption(enc)

    return caption

