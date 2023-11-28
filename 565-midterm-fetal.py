# FETAL FRONTEND
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
from matplotlib import colors
import warnings
import pandas.io.formats.style
from matplotlib import colormaps

st.title("Classifying Fetal Health: A Machine Learning App")
st.image('fetal_health_image.gif')

# Decision Tree Model
dt_pickle = open('dt_fetal.pickle', 'rb') 
dt_model = pickle.load(dt_pickle) 
dt_pickle.close()

# Random Forest Model
rf_pickle = open('rf_fetal.pickle', 'rb') 
rf_model = pickle.load(rf_pickle) 
rf_pickle.close()

# Loading default data
default_df = pd.read_csv('fetal_health.csv')


user_fetal_file = st.file_uploader('Upload your own data to predict fetal health: ')
st.write('Please ensure your data is in the following form: ')

rubricT = default_df.drop(columns=['fetal_health']).head(0)
st.write(rubricT)

if user_fetal_file is not None:
    user_df = pd.read_csv(user_fetal_file)
    original_df = default_df.copy()

    #dropping null values
    user_df = user_df.dropna()
    original_df = original_df.dropna()

    original_df = original_df.drop(columns=['fetal_health'])

    user_df = user_df[original_df.columns]

    user_pred = rf_model.predict(user_df)

    user_pred_prob = rf_model.predict_proba(user_df)

    user_df['Predicted Classification'] = user_pred

    user_df['Predicted Probability'] = user_pred_prob.max(axis=1)

    df2 = user_df.copy()
    
 

    def normal_color(val):
        if val == 1:
            color = 'lime'
        return ['color: %s' % color in user_df['Predicted Classification']]
    
    user_df.style.applymap(normal_color)
    #user_df.style.background_gradient(cmap= 'summer', user_df['Predicted Classification'])
    #23
    #user_df = user_df.style.applymap(class_color,user_df['Predicted Classification'].values)
    #user_df['Predicted Classification'].style.apply(normal_color)

    st.subheader('Predicting User-Provided Fetal Health:')
    st.write(user_df)
    #st.dataframe(df2)
    st.subheader("Prediction Performance")
    tab1, tab2, tab3 = st.tabs([ "Feature Importance", "Confusion Matrix", "Classification Report"])

    with tab1:
        st.image('rf_feature_imp.svg')
    with tab2:
        st.image('rf_confusion_mat.svg')
    with tab3:
        df = pd.read_csv('rf_class_report.csv', index_col=0)
        st.dataframe(df)



