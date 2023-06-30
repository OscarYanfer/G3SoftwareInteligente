import streamlit as st
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler         
from sklearn.model_selection import train_test_split      
from sklearn.linear_model import LinearRegression         
from sklearn.metrics import mean_squared_error,mean_absolute_error
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import seaborn as sns
import missingno as msno 
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
#cargar el archivo
#Sección 2
st.markdown("# Sección III - Exploración de modelos")
uploaded_file = st.file_uploader("Cargar archivo Excel:", type=["xlsx"])
if uploaded_file is not None:
    #Columna iniciasl será el index
    st.write("""Conjunto de datos cargado: """)
    df = pd.read_excel(uploaded_file, sheet_name='Hoja1', index_col=0)
    st.dataframe(df)
    
   
    
    seed = 4353
    data = df
    data.columns = data.columns.str.lower()
    data.isnull().sum()
    data.dropna(how='all',inplace=True)
    data.dropna(thresh=10,axis=0,inplace=True)
    data=data.drop([ 'año', 'mes', 'dia', 'hora'],axis=1)

    ###Interactivo
    st.title("Comparación de modelos de Machine Learning")
    show_linear_regression = st.checkbox("Mostrar modelo de Regresión Lineal")
    show_random_forest = st.checkbox("Mostrar modelo de Random Forest")
    show_decision_tree = st.checkbox("Mostrar modelo de Árbol de Decisiones")
    show_support_vector_machine = st.checkbox("Mostrar modelo de Support Vector Machine")

    col_ = data.columns.tolist()[2:]
    X = data[col_].drop('pm2.5 \n(ug/m3)', axis=1)
    y = data['pm2.5 \n(ug/m3)']
    X = X.apply(pd.to_numeric, errors='coerce')
    y = y.apply(pd.to_numeric, errors='coerce')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = []
    models = []
    if show_linear_regression:
        lr = LinearRegression()
        lr_model = lr.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append(rmse)
        models.append("MLR")

    if show_random_forest:
        rf_reg = RandomForestRegressor()
        rf_model = rf_reg.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        rmseRF = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        results.append(rmseRF)
        models.append("Random Forest")

    if show_decision_tree:
        dt_one_reg = DecisionTreeRegressor()
        dt_model = dt_one_reg.fit(X_train, y_train)
        y_pred_dtone = dt_model.predict(X_test)
        rmseMt = np.sqrt(mean_squared_error(y_pred_dtone, y_test))
        results.append(rmseMt)
        models.append("Tree Regression")

    if show_support_vector_machine:
        sv_reg = SVR()
        sv_model = sv_reg.fit(X_train, y_train)
        y_pred_sv = sv_model.predict(X_test)
        rmseSVM = np.sqrt(mean_squared_error(y_test, y_pred_sv))
        results.append(rmseSVM)
        models.append("SVM")

    if results:
        st.write("Comparando los modelos según el RMSE...")
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))
        plt.yticks(np.arange(0, 100, 10))
        plt.ylabel("RMSE")
        plt.xlabel("Modelos de Machine Learning")
        sns.barplot(x=models, y=results)
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.write("Selecciona al menos un modelo para mostrar.")
