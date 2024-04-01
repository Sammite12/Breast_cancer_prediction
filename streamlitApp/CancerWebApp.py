# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import dump, load
from lime import lime_tabular
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import scikitplot as skplt

import warnings
warnings.filterwarnings("ignore")

# load data
df = pd.read_csv("cleaned_df4.csv")

X =  df.iloc[:,:-1].values
y = df.iloc[:,30].values
feature_name = df.drop("diagnosis",axis=1).columns

target_name = df.diagnosis.unique()

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# load model
rf_model = load("cancer_rf3.model")
y_pred = rf_model.predict(x_test)

# DashBoard
st.title("Breast Cancer :red[Prediction] :bar_chart: :chart_with_upwards_trend:")
st.markdown("Predict Malignant or Benign Tissue using Biopsy features :hospital:")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html = True)


tab1, tab2, tab3 = st.tabs(["Data :clipboard:","Global Performance :weight_lifter:", "Local Performance :rocket:"])
with tab1:
    st.header("Breast Cancer Dataset")
    st.write(df)

with tab2:
    st.header("Confusion Matrix | Features Importances")
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure(figsize = [6,6])
        ax1 = fig.add_subplot(111)
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, ax = ax1)
        st.pyplot(fig, use_container_width = True)

    with col2:
        fig2 = plt.figure(figsize = [6,6])
        ax1 = fig2.add_subplot(111)
        skplt.estimators.plot_feature_importances(rf_model, feature_names = feature_name,  ax = ax1,x_tick_rotation = 90)
        st.pyplot(fig2, use_container_width = True)

    st.divider()
    st.header("Classification Report")
    st.code(classification_report(y_test,y_pred))

with tab3:
    st.subheader("Predict and View Model Confidence")
    sliders = []
    col1, col2 = st.columns(2, gap = "medium")
    with col1:
        for item in list(feature_name):
            feature_slider = st.slider(label = item, min_value = float(df[item].min()), max_value = float(df[item].max()))
            sliders.append(feature_slider)
    
    with col2:
        col1, col2 = st.columns(2, gap = "large")

        prediction = rf_model.predict([sliders])
        with col1:        
            st.markdown("#### Model Prediction : {}".format(target_name[prediction[0]]))
            st.write("Where 0 represent Malignant")

        probs = rf_model.predict_proba([sliders])
        probability = probs[0][prediction[0]]

        with col2:    
            st.metric(label = "Model Confidence", value = "{:.2f} %".format(probability*100), delta = "{:.2f} %".format((probability-0.6)*100))
        

        interpretor = lime_tabular.LimeTabularExplainer(x_train, mode = "classification",  class_names = target_name, feature_names = feature_name)
        
        explanation = interpretor.explain_instance(np.array(sliders), rf_model.predict_proba,num_features = 20,top_labels=2)

        interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
        st.pyplot(interpretation_fig, use_container_width= True)
    
