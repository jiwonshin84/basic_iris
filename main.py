import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title='Simple Prediction App',
    page_icon='🌷',
    layout='wide',
    initial_sidebar_state='expanded'
)

# title of app
st.title('🌷 Simple Prediction App')

df = pd.read_csv('https://raw.githubusercontent.com/jiwonshin84/basic_iris/refs/heads/main/Data/Iris.csv')
st.write(df)

# input widgets
st.sidebar.subheader('Input Features')
sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)
