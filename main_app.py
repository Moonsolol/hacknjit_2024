import streamlit as st

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

def main():
    st.title('HackNJIT 2024 Project')
    st.subheader('by Terry Su')
    st.markdown(
        """
        
        <br><br/>
        Test

        """
    )