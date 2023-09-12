import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
import pickle


# background image
background_style = """
    <style>
        body {
            background-image: url('china_growth.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
    </style>
"""
st.markdown(background_style, unsafe_allow_html=True)
st.image('china_growth.jpg', use_column_width=True)

# scaler and model
chinese_gdp_forecast_scaler = pickle.load(open('chinese_gdp_forecast_scaler.pkl', 'rb'))
chinese_gdp_forecast_model = pickle.load(open('chinese_gdp_forecast_model.pkl', 'rb'))

# "structure"
st.title('China Growth Forecast')

with st.form(key='form_parameters'):
   PMI = st.slider('PMI', 40.0, 60.0, 50.0)
   electricity = st.slider('Electricity output (100 million KWH)', 5000, 15000, 5000)
   freight_traffic = st.slider('Freight traffic (10000 tons)', 300000, 800000, 300000)
   st.markdown('---')

   submitted = st.form_submit_button('Predict')

# data adding
data_inf = {
   'PMI': PMI,
   'Output of Electricity Current Period(100 million kwh)': electricity,
   'Freight Traffic Current Period(10000 tons)': freight_traffic
}

data_inf = pd.DataFrame([data_inf])

if submitted:
   data_inf_sc = chinese_gdp_forecast_scaler.transform(data_inf)
   y_pred = chinese_gdp_forecast_model.predict(data_inf_sc)
   st.write('Gross Domestic Product (100 million yuan) = '+ str(y_pred))

