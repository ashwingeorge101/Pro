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

#st.set_page_config(page_title='demo',page_icon=':bar_chart:',layout='wide')
st.title("POWERTELL")

data=st.file_uploader('upload a file')
df=pd.read_csv(data)
df["utc_timestamp"] = df["utc_timestamp"].astype(np.datetime64)  # set the data type of the datetime column to np.datetime64
df.set_index("utc_timestamp", inplace=True)  # set the datetime columns to be the index
df.index.name = "datetime"  # change the name of the index
df=df.fillna(0)
country=option_menu('Country',['None','Austria', 'Deutschland', 'Deutschland_50hertz', 'Deutschland_LU', 'Deutschland_amprion', 'Deutschland_tennet', 'Deutschland_transnetbw', 'Hungary', 'Netherlands'], default_index=0)
if (country == 'None'):
    st.warning('Please select a country')

#1
if (country == 'Austria'):
    df=df.iloc[:,3:5]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["AT_price_day_ahead","AT_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    n_features = 2
    X_train, y_train = split_series(train_df.values,n_past, n_future)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
    X_test, y_test = split_series(test_df.values,n_past, n_future)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
    news=tf.keras.models.load_model("presentation4.h5")
    news.summary()
    pred_e1d1=news.predict(X_test)

    for index,i in enumerate(train_df.columns):
        scaler = scalers['scaler_'+i]
        pred_e1d1[:,:,index]=scaler.inverse_transform(pred_e1d1[:,:,index])
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
       a=pred_e1d1[i][0][1]
       c.append(a)  
    c = [0 if ele<0 else ele for ele in c]
    st.success('Success message')
    test_data = test_data.reset_index()
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.AT_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.AT_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)
    # d=[]
    # #for i in pred_e1d1:
    #   #for j in range(0,1):
    #     #a=pred_e1d1[0][j][6]
    #   #c.append(a)
    # for i in range(0,x):
    #    a=pred_e1d1[i][0][0]
    #    d.append(a)  
    # d = [0 if ele<0 else ele for ele in d]
    # st.success('Success message')
    # #test_data = test_data.reset_index()
    # sns.set_context("poster")
    # # plot the ground-truth and forecast and compare them with residuals
    # sns.set_context("poster")
    # fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    # sns.lineplot(x = test_data.datetime[743:], y=test_data.AT_price_day_ahead[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    # sns.lineplot(x = test_data.datetime[743:], y=d, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # # set the axis labels and title
    # ax3.set_xlabel("Date")
    # ax3.set_ylabel("Solar Generation (MW)")
    # ax3.set_title("Time Series Data");  
    # #plot for residual
    # residuals = (test_data.AT_price_day_ahead[743:]- d)
    # sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    # ax4.set_ylabel("Residuals"); # set the y-label for residuals
    # st.pyplot(fig)
    f=[]
    for i in range(0,x):
      a=pred_e1d1[i][0][0]
      f.append(a)  
    f = [0 if ele<0 else ele for ele in f]
     # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3) = plt.subplots(1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    #sns.lineplot(x = test_data.datetime[20000:39500], y=test_data.AT_price_day_ahead[20000:30000], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[:39500], y=f[:39500], color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar price prediction (euro)")
    ax3.set_title("Time Series Data");  
    st.pyplot(fig)

#2
if (country == 'Deutschland'):
    df=df.iloc[:,11:12]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["DE_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.DE_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.DE_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)

#3    
if (country == 'Deutschland_50hertz'):
    df=df.iloc[:,24:25]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["DE_50hertz_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.DE_50hertz_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.DE_50hertz_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)

#4    
if (country == 'Deutschland_LU'):
    df=df.iloc[:,30:31]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["DE_LU_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.DE_LU_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.DE_LU_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)


#5
if (country == 'Deutschland_amprion'):
    df=df.iloc[:,36:37]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["DE_amprion_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.DE_amprion_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.DE_amprion_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)


#6
if (country == 'Deutschland_tennet'):
    df=df.iloc[:,40:41]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["DE_tennet_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.DE_tennet_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.DE_tennet_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)

#7
if (country == 'Deutschland_transnetbw'):
    df=df.iloc[:,46:47]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["DE_transnetbw_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.DE_transnetbw_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.DE_transnetbw_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)


#8
if (country == 'Hungary'):
    df=df.iloc[:,50:51]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["HU_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.HU_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.HU_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)

#9
if (country == 'Netherlands'):
    df=df.iloc[:,56:57]
    a=round(len(df.index)*0.2)
    b = a
    test_data=df[["NL_solar_generation_actual"]].copy()
    test_data=test_data[-a:]
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
    sns.set_context("poster")
    # plot the ground-truth and forecast and compare them with residuals
    sns.set_context("poster")
    fig,(ax3,ax4) = plt.subplots(2,1, figsize=(50, 20), sharex=True) # get the figure dimensions for the two figures and plot on the same x-axis
    sns.lineplot(x = test_data.datetime[743:], y=test_data.NL_solar_generation_actual[743:], color="red", ax=ax3, label="original") # get the ground-truth validation data
    sns.lineplot(x = test_data.datetime[743:], y=c, color="green", dashes=True, ax=ax3, label="Forecast", alpha=0.5)  # get the forecast
    # set the axis labels and title
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Solar Generation (MW)")
    ax3.set_title("Time Series Data");  
    #plot for residual
    residuals = (test_data.NL_solar_generation_actual[743:]- c)
    sns.lineplot(y=residuals, x=test_data.datetime[743:], ax=ax4, label="Residuals")
    ax4.set_ylabel("Residuals"); # set the y-label for residuals
    st.pyplot(fig)