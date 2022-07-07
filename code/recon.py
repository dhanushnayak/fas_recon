import pickle
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import pandas as pd

base_dir='/'.join(os.path.abspath(os.path.dirname(__file__)).split('\\')[:-1])
print('/'.join(os.path.abspath(os.path.dirname(__file__)).split('\\')[:-1]))
path = base_dir+'/assets'
if not os.path.exists(path):
    path = base_dir+"/app/assets"
    base_dir = "/app"
feature_list = np.array(pickle.load(open(path+"/features.pkl",'rb')))

files_name = np.array(pickle.load(open(path+"/filenames.pkl",'rb')))

model = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainble=False

final_model = Sequential(
    [model,GlobalMaxPooling2D()]
)


def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,0)
    preprocess_img = preprocess_input(expanded_img)
    result = model.predict([preprocess_img]).flatten()
    result = result/np.linalg.norm(result)
    return result


def get_output(filename):
    feature_of_inp = extract_features(filename,final_model)
    reco_model = NearestNeighbors(n_neighbors=6,algorithm="brute",metric='euclidean')
    reco_model.fit(feature_list)
    distance,indices = reco_model.kneighbors([feature_of_inp])
    df = pd.Series(files_name).iloc[indices[0]].values
    images = []
    for i,j in enumerate(df): 
        image_path = base_dir+'/'+j
        image_path = image_path.replace('\\','/')
        if i<4: images.append(Image.open(image_path))
    return images

    


