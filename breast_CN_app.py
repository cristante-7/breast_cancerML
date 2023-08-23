import streamlit as st
import numpy as np
import pickle
import altair as alt
from streamlit_option_menu import option_menu


X = np.load(r"C:\Users\Chris\Desktop\BreastCancerML_Folder\breast-cancer-diagnosis-ml\src\features\feature_matrix.npy")


#loading the saved model

brCancer_model = pickle.load(open(r"C:\Users\Chris\Desktop\BreastCancerML_Folder\breast-cancer-diagnosis-ml\models\svm_model.sav","rb"))

# Streamlit app
st.title("Breast Cancer Prediction")

# Sidebar input
st.sidebar.header("User Input")
radius_mean = st.sidebar.slider("Radium Mean", float(np.min(X[:, 0])), float(np.max(X[:, 0])), float(np.mean(X[:, 0])))
texture_mean = st.sidebar.slider("Texture Mean", float(np.min(X[:, 1])), float(np.max(X[:, 1])), float(np.mean(X[:, 1])))
concavity_mean = st.sidebar.slider("Concavity Mean", float(np.min(X[:, 1])), float(np.max(X[:, 1])), float(np.mean(X[:, 1])))
concave_point_mean = st.sidebar.slider("Concave point Mean", float(np.min(X[:, 1])), float(np.max(X[:, 1])), float(np.mean(X[:, 1])))
area_mean = st.sidebar.slider("Area Mean", float(np.min(X[:, 1])), float(np.max(X[:, 1])), float(np.mean(X[:, 1])))
perimeter_mean = st.sidebar.slider("Perimeter Mean", float(np.min(X[:, 1])), float(np.max(X[:, 1])), float(np.mean(X[:, 1])))
compactness_mean = st.sidebar.slider("Compactness Mean", float(np.min(X[:, 1])), float(np.max(X[:, 1])), float(np.mean(X[:, 1])))


# Predict button
if st.sidebar.button("Predict"):
    input_data =  np.array([[radius_mean, texture_mean,concavity_mean,concave_point_mean,area_mean,perimeter_mean,compactness_mean]])
    pred = brCancer_model.predict(input_data)[0]
    pred_prob = brCancer_model.predict_proba(input_data)[0]
    
    if pred == 0:
        st.write("There is a {:.2f}% chance that this patient has benign cancer (Cancerous).".format(pred_prob[0] * 100))
    else:
        st.write("There is a {:.2f}% chance that this patient has malignant cancer (Non Cancerous).".format(pred_prob[1] * 100))
