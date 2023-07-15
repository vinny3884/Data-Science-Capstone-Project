# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:38:33 2023

@author: Vineet
"""


import streamlit as st

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

model=joblib.load("best_model.joblib")


def main():
    # Set the app title
    st.title('Car Price Predictor :car:')

    # Create input widgets for user inputs
    Model_name = st.text_input('Input model name',"Type here")
    year = st.text_input('Input Year',"Enter year of purchase")
    km_driven = st.number_input('Kilometers Driven', min_value=1, max_value=1000000, value=50000)
    fuel = st.selectbox('Choose Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG','Electric'])
    seller_type = st.selectbox('Choose Seller Type', ['Individual', 'Dealer','Trustmark Dealer'])
    transmission = st.radio('Select Transmission', ['Manual', 'Automatic'])
    owner = st.select_slider('Chooose type of Ownership', ['Test Drive Car','First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
    
    # Split car name into model and manufacturer
    #car_parts = Model_name.split(' ', 1)
    #manufacturer = car_parts[0] if len(car_parts) >= 1 else ''
    #model = car_parts[1] if len(car_parts) >= 2 else ''

    # Create a feature vector from the user inputs
    user_input = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel_Diesel': [1 if fuel == 'Diesel' else 0],
        'fuel_LPG': [1 if fuel == 'LPG' else 0],
        'fuel_Petrol': [1 if fuel == 'Petrol' else 0],
        'fuel_CNG': [1 if fuel == 'CNG' else 0],
        'seller_type_Individual': [1 if seller_type == 'Individual' else 0],
        'seller_type_Dealer': [1 if seller_type == 'Dealer' else 0],
        'transmission_Manual': [1 if transmission == 'Manual' else 0],
        'owner_Second Owner': [1 if owner == 'Second Owner' else 0],
        'owner_Third Owner': [1 if owner == 'Third Owner' else 0],
        'owner_Fourth & Above Owner': [1 if owner == 'Fourth & Above Owner' else 0],
        'owner_Test Drive Car': [1 if owner == 'Test Drive Car' else 0],
        #'manufacturer': [manufacturer],
        #'model': [model]
    })
    
    # Encode the 'model' and 'manufacturer' columns
    #user_input_encoded = pd.get_dummies(user_input, columns=['model', 'manufacturer'], drop_first=True)


    if st.button('Predict'):
        # Preprocess the user input features
        user_input[['year', 'km_driven']] = scaler.fit_transform(user_input[['year', 'km_driven']])


        # Make predictions using the loaded model
        predicted_price = model.predict(user_input)

        # Display the predicted price to the user
        st.subheader('Predicted Selling Price')
        st.write(f"{predicted_price[0]:.2f}")
        
    # Generate and display a bar chart
    data = {
        'Model': ['Lin_Reg', 'Lasso_Reg', 'Ridge_Reg', 'Decision Tree', 'KNN', 'SVM', 'Random Forest', 'Ada boost', 'XG boost'],
        'R^2 Score': [0.414878, -0.000076, 0.414836, 0.006658, 0.416509, 0.477010, 0.524149, 0.285312, 0.364411]
    }
    fig, ax = plt.subplots()
    ax.bar(data['Model'], data['R^2 Score'])
    ax.set_ylabel('R^2 Score')
    ax.set_xlabel('Model')
    ax.set_title('Model Performance')
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
    

if __name__ == '__main__':
    main()










