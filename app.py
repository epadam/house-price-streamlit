import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.special import boxcox
# import tensorflow as tf
# from tensorflow import keras
# from keras import models

title = st.title('House price prediction (a regression problem)')
st.image('housesbanner.png', use_column_width = True)
Description = st.write('In this demoe, the task is to predict the house price. dataset is kaggle house price dataset and sklearn. We will use machine learning models in scikit learn and Xgboost to predict the house price.')

st.write('this is an interactive demo that you can process the data, train, evaluate and explore the model')


#st.sidebar.write('Steps from preprocessing, training, evaluation to model exploring')

st.text('1. Data preprocessing')
da = st.button('Data analysis and visualization')
preprocessing = st.button('Show the preprocessing results')

#remove the button all change to selectbox to keep the value


@st.cache
def data_preprocessing():
    # maybe let the user try preprocesinng themselves
    
    train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    
    
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features = all_features.fillna(all_features.mean())
    
    all_features = pd.get_dummies(all_features, dummy_na=True)
    
    
    n_train = train_data.shape[0]
    train_features = np.array(all_features[:n_train].values)
    test_features = np.array(all_features[n_train:].values)
    train_labels = np.array(train_data.SalePrice.values).reshape((-1, 1))
    

    return train_data, test_data, all_features, train_features, test_features, train_labels

train_data, test_data, all_features, train_features, test_features, train_labels = data_preprocessing()

@st.cache
def model_training(model, parameters, x, y):
    pass
    # model = model(parameters)
    # model.fit(x, y)
    #return

import seaborn as sns
import matplotlib.pyplot as plt


def draw_corr_picture(X):   # 绘制相关系数图
    corrmat = X.corr()
    fig, axis = plt.subplots(figsize=(12,12))
    sns.heatmap(corrmat,vmax=0.9,square=True,cmap='Blues')
    return fig
    
def draw_sale_picture(X):   # 绘制相关系数图
    fig, axis = plt.subplots(figsize=(12,12))
    sns.distplot(X.SalePrice)
    return fig





if da:
    fig = draw_corr_picture(train_data)
    st.write('Correlation between features')
    st.pyplot(fig)
    st.write('Distribution of sale price')
    fig = draw_sale_picture(train_data)
    st.pyplot(fig)


    



if preprocessing:
    
    st.subheader('Total 80 features and 1 label(Sale Price)')
    data1 = st.write(train_data.iloc[0])
    st.subheader('numeric data standerization and fill missing value with mean')
    data2 = st.write(all_features.iloc[0])
    st.subheader('discrete values to one-hot encoding')
    data3 = st.write(all_features.iloc[0])

    
algorithm = st.selectbox('2. Pick a machine learning model',
    ('Chooese a model', 'Linear Regression', 'SVR', 'Random Forest Regression', 'XGboost'))

    

if algorithm == 'Linear Regression':
    # model = linearregerssion()
 
    st.header('linear Regression')
    st.subheader('training options')
    st.checkbox('fit_intercept')
    tra = st.button('Train the model')
    if tra:
        # accuracy, results, model= model_training(model, parameters, x, y)
        # st.write(accuracy)
        pass    
    xAI = st.selectbox('3. Explore the model', ('Permutation Importantce', 'Partial Dependence Plot', 'SHAP'))
    # accuracy, results, model= model_training(model, parameters, x, y)
    if xAI == 'SHAP':
        pass

    


    

    






if algorithm == 'SVR':
    train_data, test_data, all_features, train_features, test_features, train_labels = data_preprocessing()

    st.header('SVR')
    
    st.subheader('training options')
 
    kernal= st.radio(
        'Select kernel:',
        ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))
    degree = st.slider('degree', 1, 2, 3)

    train = st.button('Train the model')

    # shoe the results
   


    








#st.sidebar.button('compare model performance')



















