from numpy import split
import streamlit as st
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
import matplotlib.pyplot as plt  # data-visualization
from streamlit_option_menu import option_menu
#%matplotlib inline
import seaborn as sns  # built on top of matplotlib
sns.set()
import pandas as pd  # working with data frames
import plotly.express as px
import numpy as np  # scientific computing
import missingno as msno  # analysing missing data
#import tensorflow as tf  # used to train deep neural network architectures
import tensorflow as tf  # used to train deep neural network architectures
from tensorflow.python.keras import layers
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.keras.models import load_model
st.set_page_config(page_title='act',page_icon=':bar_chart:',layout='wide')
st.title("POWERTELL")
data=st.file_uploader('upload a file')
df=pd.read_csv(data)

d=st.number_input("Enter the datatime column number")
d=round(d)
j=st.number_input("Enter the solar production column number")
j=round(j)
#country=st.text_input("Enter the country name ")
time=st.number_input("Time interval (Hours) between samples")
price=st.number_input("Enter the energy price(in USD) per MWh of the region in your dataset")
name=df.columns[d]
solar=df.columns[j]
st.markdown(solar)
df=df.fillna(0)
df[name]=pd.to_datetime(df[name])
#df[name]=df[name].astype(np.datetime64)
df.set_index(name, inplace=True)  # set the datetime columns to be the index
df.index.name = "datetime"  # change the name of the index
#st.dataframe(df)
df=df.iloc[:,j-1:j]
st.dataframe(df)
a=round(len(df.index)*0.2)
b = a
#test_data=df[[solar]].copy()
test_data=df.copy()
test_data=test_data[-a:]
st.dataframe(test_data)
train_df,test_df=df[:-a],df[-a:]
st.success('Success message')
train = train_df
scalers={}
for i in train_df.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+ i] = scaler
    train[i]=s_s
test = test_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    s_s=np.reshape(s_s,len(s_s))
    scalers['scaler_'+i] = scaler
    test[i]=s_s
def split_series(series, n_past, n_future):
  
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)

n_past = 720
n_future =24
n_features = 1
X_train, y_train = split_series(train_df.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_test, y_test = split_series(test_df.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
news=tf.keras.models.load_model("presentation5.h5")
news.summary()
pred_e1d1=news.predict(X_test)

for index,i in enumerate(train_df.columns):
    scaler = scalers['scaler_'+i]
    pred_e1d1[:,:,index]=scaler.inverse_transform(pred_e1d1[:,:,index])
    #pred_e2d2[:,:,index]=scaler.inverse_transform(pred_e2d2[:,:,index])
    y_train[:,:,index]=scaler.inverse_transform(y_train[:,:,index])
    y_test[:,:,index]=scaler.inverse_transform(y_test[:,:,index])

from sklearn.metrics import mean_absolute_error
for index,i in enumerate(train_df.columns):
  print(i)
  for j in range(1,2):
    print("Hour ",j,":")
    print("MAE-E1D1 : ",mean_absolute_error(y_test[:,j-1,index],pred_e1d1[:,j-1,index]))
  print()
x = len(pred_e1d1)
c=[]
#for i in pred_e1d1:
#for j in range(0,1):
#a=pred_e1d1[0][j][6]
#c.append(a)
for i in range(0,x):
   a=pred_e1d1[i][0][0]
   c.append(a)  
c = [0 if ele<0 else ele for ele in c]
st.success('Success message')
test_data = test_data.reset_index()
st.markdown(solar)
new=test_data.iloc[767:,1]
sns.set_context("poster")
# plot the ground-truth and forecast and compare them with residuals
sns.set_context("poster")
fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
sns.lineplot(x = test_data.datetime[767:], y=new, color="red", ax=ax3, label="original") # get the ground-truth validation data
sns.lineplot(x = test_data.datetime[767:], y=c[:-24], color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
# set the axis labels and title
ax3.set_xlabel("Date")
ax3.set_ylabel("Solar Generation (MW)")
ax3.set_title("Time Series Data");  
#plot for residual
residuals = (new- c[:-24])
sns.lineplot(y=residuals, x=test_data.datetime[767:], ax=ax4, label="Residuals")
ax4.set_ylabel("Residuals"); # set the y-label for residuals
st.pyplot(fig)
distribution=sum(c[-24:])
price_prediction=distribution*price
hour=time*24
#prediction=price_prediction*hour
st.write('We are going to predict 24 samples. Since your sample size is {} in hours, and the cost of electricity(in USD) {} per MWh, you would get a profit of {} USD  in {} hours if you employ solar power units to harvest energy.'.format(time,price,price_prediction,hour))