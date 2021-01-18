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
from sklearn.linear_model import LinearRegression


train_data = pd.read_csv('/home/andrew/Documents/project/streamlit-app/houseprice/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/home/andrew/Documents/project/streamlit-app/houseprice/house-prices-advanced-regression-techniques/test.csv')


all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features = all_features.fillna(all_features.mean())
    
all_features = pd.get_dummies(all_features, dummy_na=True)
    

n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values)
test_features = np.array(all_features[n_train:].values)
train_labels = np.array(train_data.SalePrice.values)
#.reshape((-1, 1))

LR = LinearRegression()
LR.fit(train_features, train_labels)
print(LR.predict(test_features))


XGB = XGBRegressor()
XGB.fit(train_features, train_labels)
print(XGB.predict(test_features))
#from sklearn.metrics import scorer
import eli5
from eli5.sklearn import PermutationImportance
from eli5.sklearn.metrics import scorer

perm = PermutationImportance(LR, random_state =1).fit(train_features, train_labels)
eli5.show_weights(perm, feature_names = train_features.tolist())