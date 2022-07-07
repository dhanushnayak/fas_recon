from fileinput import filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainble=False

model = Sequential(
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

    

filesname = []
for file in os.listdir('../images/images'):
    filesname.append(os.path.join('images/images',file))


feature_list  = []
for file in tqdm(filesname):
    result = extract_features(file,model)
    feature_list.append(result)


pickle.dump(feature_list,open("../assets/features.pkl",'wb'))
if os.path.exists('../assets/features.pkl'):
    print("Feature file extracted")
pickle.dump(filesname,open("../assets/filenames.pkl",'wb'))
if os.path.exists('../assets/filenames.pkl'):
    print("Filenames file extracted")
