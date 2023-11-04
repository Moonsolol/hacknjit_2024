import streamlit as st

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

st.set_page_config(
    layout="centered", 
    page_title="HackNJIT 2024 Project"
)

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

st.sidebar.write("")
st.sidebar.button("Reset", type="primary")

MainTab, AboutTab = st.tabs(["Main", "About"])

with AboutTab:
    st.title('HackNJIT 2024 Project')
    st.subheader('by Terry Su')
    st.markdown(
        """
        
        <br><br/>
        Test

        """
    )

with MainTab:
    st.title('HackNJIT 2024 Project - Main Page')
    if st.sidebar.button('Train model from scratch'):
        #Get the images paths and labels and load them into a dataframe
        images = []
        labels = []
        dataset_path = "dataset"
        folders = os.listdir(dataset_path)
        for i in folders:
            fpath = os.path.join(dataset_path, i)
            files = os.listdir(fpath)
            for file in files:
                ipath = os.path.join(fpath, file)
                images.append(ipath)
                labels.append(i)
        df = pd.concat([pd.Series(images, name= 'images'), pd.Series(labels, name= 'labels')], axis=1)
        lbl = df['labels']

        #Split the dataframe into train, test and validation sets
        train_df, temp_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42, stratify=lbl)
        lbl = temp_df['labels']
        valid_df, test_df = train_test_split(temp_df, train_size=0.5, shuffle=True, random_state=42, stratify=lbl)

        #Fits the training dataframe into an image data generation to augment the dataset
        img_shape = (150, 150, 3)

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.2,1.2]
        )
        traingen = datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='images',
            y_col='labels',
            target_size=(150,150),
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True,
            batch_size=15
        )

        gendict = traingen.class_indices
        classes = list(gendict.keys())
        imgs, lbls = next(traingen)
        plt.figure(figsize=(20,20))
        for i in range(15):
            plt.subplot(5, 5, i + 1)
            image = imgs[i]/255
            plt.imshow(image)
            index = np.argmax(lbls[i])
            class_name=classes[index]
            plt.title(class_name, color='blue',fontsize=12)
            plt.axis='off'
        st.write('Sample of Augmented Dataset with Labels')
        st.pyplot(plt)

    else:
        st.write("Choose an option from the sidebar")
