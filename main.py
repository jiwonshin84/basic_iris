import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

df.columns= [ col_name.split('Cm')[0] for col_name in df.columns] # 컬럼명을 뒤에 cm 제거

st.write(df)

df = df.drop('Id', axis=1)

st.write(df)

# input widgets
st.sidebar.subheader('Input Features')
sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)


# 예측 모델 생성
# Separate X and y
X = df.drop('Species', axis=1)
y = df.Species

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
rf = RandomForestClassifier(max_depth=2, max_features=4, n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Apply Model to make predictions
y_pred = rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write(y_pred)

# 간단한 EDA_ 아이리스 종에 따른 4개 컬럼 평균 계산

# print EDA
st.subheader('Brief EDA')
st.write('The data is grouped by the class and the variable mean is computed for each class.')
groupby_species_mean = df.groupby('Species').mean()
st.write(groupby_species_mean)

# 슬라이더에 Input 한 컬럼 값을 데이터프레임으로 출력하기
# print input Features
input_feature = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
st.write(input_feature)

# 예측 결과값을 metric으로 출력
st.subheader('Output')
st.metric('Predicted class', y_pred[0], '')

# 🔮 예측
y_proba = rf.predict_proba(input_feature)


# 📈 예측 확률 출력
st.subheader("Prediction Probabilities")
prob_df = pd.DataFrame(data=y_proba, columns=rf.classes_)
st.write(prob_df)

# 📊 Confusion Matrix
st.subheader("Confusion Matrix (on Test Set)")

y_test_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred, labels=rf.classes_)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

