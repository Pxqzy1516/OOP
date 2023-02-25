import streamlit as st
from PIL import Image
from streamlit_extras.stoggle import stoggle

imcup = Image.open('cup.png')
st.set_page_config(
    page_title='CAFFEINE4TODAY',
    page_icon=(imcup),
    layout='centered')

st.title('Code for this web')
code = '''from sklearn.linear_model import LogisticRegression
import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import time as t

def load_data():
    return pd.read_csv('caffeine.csv')

def save_model(model):
    joblib.dump(model, 'model.joblib')

def load_model():
    return joblib.load('model.joblib')

df = load_data()
data = pd.DataFrame(df)
data = data.drop(columns='type',axis=1)
data2 = data.drop(columns='Calories',axis=1)
#  print(data)

data1 = pd.DataFrame(data2)
X = data1.drop(columns='Caffeine (mg)')
X = X.drop(columns='drink')
y = data['Caffeine (mg)']
name = data['drink']
A = []
B = []
C = []
for i in range(len(y)):
    A.append(i)
for j in range(len(y)):
    B.append(name[j])
for k in range(len(A)):
    D = []
    D.append(A[k])
    D.append(B[k])
    C.append(tuple(D))
X['drink'] = A

x_train, x_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
model = LogisticRegression()
model.fit(x_train,y_train)
save_model(model)

imcup = Image.open('cup.png')
st.set_page_config(
    page_title='CAFFEINE4TODAY',
    page_icon=(imcup),
    layout='centered')

Menu_bar = st.progress(0)
for Menu_bar_percent in range(100):
    t.sleep(0.0000000000000000001)
    Menu_bar.progress(Menu_bar_percent)

st.title('How much Caffeine do you have today☕')
st.subheader('วันนี้รับคาเฟอีนเข้าไปเท่าไหร่แล้วนะ 👀❔')
drinks = data2[['drink']]
col1, col2 = st.columns(2)
with col1:
    DRINK = st.selectbox("Select your drink", drinks['drink'].values)
    drink_index = drinks.index[drinks['drink'] == DRINK].tolist()[0]
with col2:
    VOLUME = st.number_input('How much volume? ',min_value=1, max_value=3000)


def predict():
    model=load_model()
    data_user = (drink_index, VOLUME)
    data_array = np.array(tuple(data_user))
    data_reshape = data_array.reshape(1,-1)
    pred = model.predict(data_reshape)[0]
    st.write('The amount of caffeine is:', round(pred, 5), 'mg')

def save_data():
    df = pd.read_csv('caffeine.csv', index_col=0)
    see = pd.DataFrame(df)
    print(see)
    data.to_excel('caffeine.xlsx')
    st.write(df.head())



loadb, pre = st.columns(2)
with loadb:
    loadb = st.button('Load data')
    if loadb:
        df = pd.read_csv('caffeine.csv', index_col=0)
        X = X.drop(columns='drink')
        y = data['Caffeine (mg)']
        df = pd.DataFrame(df)
        st.dataframe(df)
with pre:
    pre = st.button("predict", predict())
'''

st.code(code,language='python')