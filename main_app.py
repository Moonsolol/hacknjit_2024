import streamlit as st

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

st.set_page_config(
    layout="centered", 
    page_title="BoatExpert"
)

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        st.write("Starting training")

    def on_train_end(self, logs=None):
        st.write("Stop training")

    def on_epoch_begin(self, epoch, logs=None):
        st.write("Start epoch {} of training".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        st.write("End epoch {} of training".format(epoch))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        st.write("Start testing")

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        st.write("Stop testing")

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        st.write("Start predicting")

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        st.write("Stop predicting")

def augment_dataset():
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
    b_size = 12

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.25,
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
        subset='training',
        target_size=(150,150),
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=b_size
    )
    validgen = datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='images',
        y_col='labels',
        subset='validation',
        target_size=(150,150),
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=b_size
    )
    testgen = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='images',
        y_col='labels',
        target_size=(150,150),
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=b_size
    )
    return traingen, validgen, testgen

def train_model(traingen, validgen):
    model = Sequential([
        Conv2D(32, 5, activation='relu', padding='same', input_shape=[150,150,3]),
        MaxPooling2D(2),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        Conv2D(128, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(9, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=0.0005),
                  metrics=['accuracy'])
    STEP_SIZE_TRAIN=traingen.n//traingen.batch_size
    STEP_SIZE_VALID=validgen.n//validgen.batch_size
    model.fit_generator(generator=traingen,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=validgen,
              validation_steps=STEP_SIZE_VALID,
              epochs=100)
    model.save("saved_model.h5")
    return model

def evaluate_model(model, traingen, validgen, testgen):
    train_score = model.evaluate(traingen, verbose=1)
    valid_score = model.evaluate(validgen, verbose=1)
    test_score = model.evaluate(testgen, verbose=1)

    st.write("Train Loss:", train_score[0])
    st.write("Train Accuracy:", train_score[1])
    st.write('-' * 20)
    st.write("Validation Loss:", valid_score[0])
    st.write("Validation Accuracy:", valid_score[1])
    st.write('-' * 20)
    st.write("Test Loss:", test_score[0])
    st.write("Test Accuracy:", test_score[1])
    st.write('-' * 20)

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

logo_url = 'images/boatexpertlogo_new.png'
st.sidebar.image(logo_url)
st.sidebar.write("")
st.sidebar.button("Reset", type="primary")

MainTab, AboutTab = st.tabs(["Main", "About"])

with AboutTab:
    st.title('BoatExpert - A HackNJIT 2024 Project')
    st.subheader('by Terry Su')
    st.markdown(
        """
        This is a boat classifier that categorizes boat images into 9 different types:
        - Buoy
        - Cruise Ship
        - Ferry Boat
        - Freight Boat
        - Gondola
        - Inflatable Boat
        - Kayak
        - Paper Boat
        - Sailboat

        The dataset used to train this model was found at: https://www.kaggle.com/datasets/clorichel/boat-types-recognition

        A sequential Keras deep learning model is trained and used to classify boat images.
        A pre-trained model is available a the link below for fast use, as it takes a long time to train a model.

        Pre-trained Model: https://drive.google.com/file/d/10xUDOr7F8P24fBgUK3gsFfQWIlUZFQ6V/view?usp=sharing
        """
    )

with MainTab:
    st.title('BoatExpert')
    uploaded_file = None
    with st.spinner('Preparing data...'):
        train_set, valid_set, test_set= augment_dataset()
    if st.sidebar.button('Train model from scratch'):
        with st.spinner('Training model... (might take a few hours)'):
            model = train_model(train_set, valid_set)
        with st.spinner('Testing model...'):
            evaluate_model(model, train_set, valid_set, test_set)
        st.success('Done!')
    elif st.sidebar.button('Use the supplied pre-trained model'):
        model = tf.keras.models.load_model('saved_model.h5')
        with st.spinner('Testing model... (may take a while, please wait)'):
            evaluate_model(model, train_set, valid_set, test_set)
        st.success('Done!')
    else:
        st.write("Choose an option from the sidebar!")
        st.write("Recommended to use the pre-trained model, training takes a very long time.")