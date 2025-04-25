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
st.title('🌷 붓꽃 데이터 실습')

df = pd.read_csv('https://raw.githubusercontent.com/jiwonshin84/basic_iris/refs/heads/main/Data/Iris.csv')
df.columns= [ col_name.split('Cm')[0] for col_name in df.columns] # 컬럼명을 뒤에 cm 제거
df = df.drop('Id', axis=1)

# 사이드바 버튼 스타일 조정
st.markdown("""
    <style>
        .stSidebar > div:first-child {
            display: flex;
            flex-direction: column;
        }
        .stSidebar button {
            height: 50px;
            width: 100%;
            font-size: 18px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# 화면 비우기용 placeholder
placeholder = st.empty()


if st.sidebar.button('🌷 붓꽃 데이터'):
    placeholder.empty()  # 이전 데이터를 지운다
    st.markdown(f"<h3 style='font-size: 24px;'>데이터 개수: {df.shape[0]}  &nbsp; / 중복 인덱스 수: {df.index.duplicated().sum()}", unsafe_allow_html=True)

    # 인덱스 리셋 후 1부터 시작하도록 변경
    df = df.reset_index(drop=True)  # 기존 인덱스를 제거하고 새로 리셋
    df.index = df.index + 1  # 인덱스를 1부터 시작하도록 변경
    
    st.write(df)

if st.sidebar.button('🎉 Brief EDA'):
    placeholder.empty()  # 이전 데이터를 지운다
    st.empty()  # 기존 EDA 결과 삭제
    
    # 간단한 EDA_ 아이리스 종에 따른 4개 컬럼 평균 계산
    # print EDA
    st.subheader('Brief EDA(간단한 탐색적 데이터 분석)')
    st.write('The data is grouped by the class and the variable mean is computed for each class.')
    groupby_species_mean = df.groupby('Species').mean()
    st.write(groupby_species_mean)


if "show_slider" not in st.session_state:
    st.session_state.show_slider = False

# 사이드바에 버튼
if st.sidebar.button("✔ 새로운 데이터 예측"):
    placeholder.empty()  # 이전 데이터를 지운다
    st.session_state.show_slider = True

if st.session_state.show_slider:
    st.empty()  # 기존 예측 결과 삭제
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
    
    st.subheader('💡 Input Features값 예측')
    
    # Apply Model to make predictions
    y_pred = rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    # st.write(y_pred)
    
    # 슬라이더에 Input 한 컬럼 값을 데이터프레임으로 출력하기
    # print input Features
    input_feature = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    st.write(input_feature)

    # 예측 결과값을 metric으로 출력
    st.metric('💡 Predicted class', y_pred[0], '')

    # 🔮 예측
    y_proba = rf.predict_proba(input_feature)
    
    # 📈 예측 확률 출력
    st.subheader("📈 예측 확률")
    prob_df = pd.DataFrame(data=y_proba, columns=rf.classes_)
    st.write(prob_df)


if st.sidebar.button('📊 Confusion Matrix'):
    st.empty()  # 기존 Confusion Matrix 삭제
    # 📊 Confusion Matrix
    st.subheader("📊 Confusion Matrix (on Test Set)")

    y_test_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred, labels=rf.classes_)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

        
        


