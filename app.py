import streamlit as st
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import keras.backend as K
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
import joblib
# from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
# import tensorflow.keras.backend as K
K.clear_session()

st.title('Multi Layer Perceptron based Predictor')

# df = pd.read_csv('./ashrae_db2.01.csv')
# data = df[['Year','Operative temperature (C)','Outdoor monthly air temperature (C)','Relative humidity (%)','Thermal sensation','Thermal comfort']]
# data.dropna(inplace=True)
# data = data.drop(data[data['Thermal comfort'] == 'Na'].index)
# X = data.drop(['Thermal sensation', 'Thermal comfort'], axis=1)
# y = data[['Thermal sensation', 'Thermal comfort']]
# y_ = y

round_ts = lambda target: min(np.array([-3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0]), key=lambda x: abs(x - target))
round_tc = lambda target: min(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), key=lambda x: abs(x - target))

# Load the Thermal Sensation model
loaded_model_ts = load_model("thermal_sensation_model.h5")

# Load the Thermal Comfort model
loaded_model_tc = load_model("thermal_comfort_model.h5")

scaler = joblib.load("scaler.pkl")
# le_ts = LabelEncoder() #thermal sensation label encoder
# le_tc = LabelEncoder() #thermal comfort label encoder

# le_ts.fit(np.array(y)[:,0])
# le_tc.fit(np.array(y)[:,1])

# y = np.array([np.array(i) for i in zip(le_ts.transform(np.array(y)[:,0]),le_tc.transform(np.array(y)[:,1]))])

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
# X_train, X_test, y_train_, y_test_ = train_test_split(X,y_,test_size=0.2, random_state=42)

# svc_ts = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# svc_tc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# svc_ts.fit(X, y[:,0])
# svc_tc.fit(X, y[:,1])


st.sidebar.header('Input Parameters')
p1 = st.sidebar.number_input(f'Operative temperature (C)', value=32.0)
p2 = st.sidebar.number_input(f'Outdoor monthly air temperature (C)', value=32.0)
p3 = st.sidebar.number_input(f'Relative humidity (%)', value=67.0)
p4 = st.sidebar.number_input(f'Air velocity (m/s)', value=0.0)
p5 = option = st.sidebar.selectbox('Season', ('Autumn', 'Spring', 'Summer', 'Winter'))
# Make predictions
#input_data = [[param1, param2, param3]]
season_code = list(['Autumn', 'Spring', 'Summer', 'Winter']).index(p5)

input_data = [[p1,p2,p3,p4,season_code]]
#input_data_scaled = scaler.transform(input_data)   
@st.cache_data
def make_prediction(ip):
    input_data = np.array(ip).reshape(1, -1)
    input_data = scaler.transform(input_data)
    return round_ts(loaded_model_ts.predict(input_data)[0][0]),  round_tc(loaded_model_tc.predict(input_data)[0][0])


prediction_ts, prediction_tc = make_prediction(input_data)
# ts = prediction_ts#le_ts.inverse_transform(svc_ts.predict(np.array(input_data[0]).reshape(1,-1)))
# tc = prediction_tc#le_tc.inverse_transform(svc_tc.predict(np.array(input_data[0]).reshape(1,-1)))

st.write('Predicted thermal sensation:', prediction_ts)
st.write('Predicted thermal comfort:', prediction_tc)
