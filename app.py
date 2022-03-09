
import os
import pathlib
os.system('wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -P .')

import random
from PIL import Image
import torch
import glob
import numpy as np
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import face_recognition

import streamlit as st
from streamlit_image_comparison import image_comparison


DIR = os.getcwd()
model_path = os.path.join(DIR, 'GFPGANCleanv1-NoCE-C2.pth')

bg_upsampler = None

restorer = GFPGANer(
    model_path='GFPGANCleanv1-NoCE-C2.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=bg_upsampler)

def enhancer(image):
    image = np.array(image)
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True)

    return Image.fromarray(restored_faces[0]), np.array(restored_img)

def inference(img1, img2):

    img1 = face_recognition.face_encodings(img1)[0]
    img2 = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1], img2)

    if result[0]:
        result = 'People in the two images are same!'
    else:
        result = 'People in the two images are different!'
    
    return result

def main():
    
    st.set_page_config(page_title='QFace.ai', page_icon='🤖', layout='centered')
    st.title('QFace.ai')
    col1, col2 = st.columns(2)
    
    with st.form('Input Form'):
        
        with col1:
            st.header('Image 1')
            img1 = st.file_uploader('Upload Image 1', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

        with col2:
            st.header('Image 2')
            img2 = st.file_uploader('Upload Image 2', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
                
        submitted = st.form_submit_button('Enhance and Compare')
        
        if submitted:
            if img1 is not None and img2 is not None:
                
                with col1:
                    img1 = Image.open(img1)
                    eimg1_face, eimg1 = enhancer(img1)
                    image_comparison(img1, eimg1_face,
                                     width=350,
                                     label1='Before',
                                     label2='After',
                                     show_labels=True,
                                     make_responsive=False)
                with col2:
                    img2 = Image.open(img2)
                    eimg2_face, eimg2 = enhancer(img2)
                    image_comparison(img2, eimg2_face,
                                     width=350,
                                     label1='Before',
                                     label2='After',
                                     show_labels=True,
                                     make_responsive=False)
                    
            result = inference(np.array(img1), np.array(img2))
            st.write(result)
    

if __name__ == '__main__':
    main()
