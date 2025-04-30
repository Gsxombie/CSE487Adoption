import streamlit as st
import joblib
import pandas as pd

rf_model = joblib.load('random_forest_model.pkl')
le_color = joblib.load('color.pkl')
le_breed = joblib.load('breed.pkl')
le_gender = joblib.load('gender.pkl')
le_animal_type = joblib.load('animal_type.pkl')

color_labels = le_color.classes_
breed_labels = le_breed.classes_
gender_labels = le_gender.classes_

def predict_adoption(color, breed, gender, animal_type):
    color_encoded = le_color.transform([color])[0]
    breed_encoded = le_breed.transform([breed])[0]
    gender_encoded = le_gender.transform([gender])[0]
    animal_type_encoded = le_animal_type.transform([animal_type])[0]
    data = pd.DataFrame([[animal_type_encoded, gender_encoded, color_encoded, breed_encoded]], columns=['animal_type', 'gender', 'primary_color', 'breed'])

    prediction = rf_model.predict(data)
    return prediction[0]

st.title('Animal Adoption Prediction')
st.write('Enter the animal details below to predict the likelihood of adoption.')
color = st.selectbox('Select the primary color:', color_labels)
breed = st.selectbox('Select the breed:', le_breed.classes_)
gender = st.selectbox('Select the gender:', gender_labels)
animal_type = st.selectbox('Select the animal type:', le_animal_type.classes_)

if st.button('Predict Adoption'):
    prediction = predict_adoption(color, breed, gender, animal_type)
    if prediction == 1:
        st.success('This animal is likely to be adopted! :smiley:')
    else:
        st.warning('This animal is not likely to be adopted :disappointed:')
