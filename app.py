
from PIL import Image
from numpy import column_stack
import streamlit as st
import code.recon as recon
import os
st.title("Fashion Recommender")
def save_uploader(uploaded):
    try: 
        with open("uploader/"+uploaded.name,'wb') as f:
            f.write(uploaded.getbuffer())
        return 1
    except:
        return 0
def get_path(uploader):
    path ="/app/uploader/"+uploader.name
    return path

def delete_cache_image(filename):
    path ="/app/uploader/"+filename.name
    if os.path.exists(path):
        os.remove(path)
    
uploader_file = st.file_uploader("Choose an image")
if uploader_file is not None:
    if save_uploader(uploader_file):
        display_img = Image.open(uploader_file)
        _,col2,_= st.columns(3)
        with col2:  
            st.image(display_img,width=100)
            with st.spinner('Processing...'):
                df = recon.get_output(get_path(uploader=uploader_file))

        with st.expander("Results"): 
            col= st.columns(len(df))
            for i in range(len(df)): 
                with col[i]:
                    st.image(df[i],width=100)
    
       
        delete_cache_image(uploader_file)    

    else:
        st.warning("Please Retry...")



