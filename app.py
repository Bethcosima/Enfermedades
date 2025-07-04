
import streamlit as st  
import joblib
import sklearn
from scipy.sparse import hstack

#cargamos los datos 
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder=  joblib.load('label_encoder.pkl')

st.title('Predicción de enfermedades a partir de síntomas')

#entrada del usuario
symptoms_input = st.text_area("Escribe tus Sintomas (en inglés)", height=200)
is_chronic = st.checkbox("los sintomas son crónicos?")
is_contagious = st.checkbox("los sintomas son contagiados?")

if st.button("Predecir enfermedad"):
  if symptoms_input.strip() == "":
    st.warning("Por favor, ingresa al menos un síntoma.")
  else:

    #Vectorizar síntomas
    X_text = vectorizer.transform([symptoms_input.lower()])
    X_extra = [[int(is_chronic), int(is_contagious)]]
    
    
    X_final = hstack([X_text, X_extra])

    pred = model.predict(X_final)
    enfermedad = label_encoder.inverse_transform(pred) [0]
    st.success(f"La enfermedad probable es: {enfermedad}**")
  
