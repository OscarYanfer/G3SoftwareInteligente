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
import base64
warnings.filterwarnings("ignore")
from io import BytesIO

#Sección I
st.markdown("# Sección I - Preprocesamiento")

#cargar el archivo
uploaded_file = st.file_uploader("Cargar archivo Excel:", type=["xlsx"])

if uploaded_file is not None:
    #Columna iniciasl será el index
    st.write("""Conjunto de datos cargado: """)
    df = pd.read_excel(uploaded_file, sheet_name='Sheet1', index_col=0)
    st.dataframe(df)
    #Eliminación de columnas con instancias únicas
    df = df.drop(['CODIGO DE LA ENTIDAD', 'CODIGO UBIGEO INEI', 'CODIGO PAIS ', 'NOMBRE DE LA UO'], axis=1)
    #Verificación de valores vacíos
    with st.spinner("Confirmación de datos vacios..."):
        time.sleep(1)
    st.write("Existencia de datos vacíos: " + str(df.isnull().any().any()))
    with st.spinner("Filtración de valores pertenecientes al primer semestre del año 2021..."):
        time.sleep(1)
    st.write("""Valores pertenecientes al primer semestre del año 2021, filtrado""")
    #Filtración de valores pertenecientes al primer semestre del año 2021
    df = df.loc[(df['Fecha_v2'] >= '2021-01-01')
                     & (df['Fecha_v2'] < '2021-07-01')]

    df_temp = df
    df_temp = df_temp.drop(['Fecha', 'Fecha_v2', 'Hora'], axis=1)
    #Guardar en un array las columnas de tipo numérico
    columnas = df_temp.columns
    with st.spinner("Calculando el promedio de cada columna numérica..."):
        time.sleep(1)
    st.write("""Promedio de cada columna numérica, calculado""")
    with st.spinner("Agregando las filas con información faltante..."):
        time.sleep(1)
    st.write("""Información faltante, agregada""")
    #Calculamos el promedio de cada columna numérica y la agregamos a las filas con información faltante e imprimimos lo que se cambiará
    for c in columnas:
        mean = df[c].mean()
        print(mean)
        df[c] = df[c].fillna(mean)

    #Verificación de valores vacíos
    with st.spinner("Confirmación de datos vacios..."):
        time.sleep(1)
    st.write("Existencia de datos vacíos: " + str(df.isnull().any().any()))

    with st.spinner("Reemplazando valores en 0 con los valores máximos de cada columna..."):
        time.sleep(1)
    st.write("""Valores en 0 con los valores máximos de cada columna, reemplazado""")
    #Reemplazar valores en 0 con los valores máximos de cada columna 
    for c in columnas:
        max = df[c].max()
        print(max)
        df[c] = df[c].replace({0: max})

    df = df.rename(columns={'Fecha':'Fecha y Hora', 'Fecha_v2':'Fecha'})
    #Transformamos la división de atributos en tipo fecha
    df['Año'] = df['Fecha'].dt.year 
    df['Mes'] = df['Fecha'].dt.month 
    df['Dia'] = df['Fecha'].dt.day 

    with st.spinner("Reordenando columnas..."):
        time.sleep(1)
    st.write("""Columnas reordenadas""")
    #Reordenamos las columnas
    df = df[['Fecha y Hora', 'Fecha', 'Año', 'Mes', 'Dia', 'Hora', 'CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)',
       'PM10 \n(ug/m3)', 'PM2.5 \n(ug/m3)', 'SO2 (ug/m3)', 'Ruido (dB)', 'UV',
       'Humedad (%)', 'Latitud', 'Longitud', 'Presion \n(Pa)',
       'Temperatura (C)']]

    #df.head()

    #Cambiando el indice a la Fecha y Hora, si es que se desea
    df.set_index("Fecha y Hora", drop=True, append=False, inplace=True, verify_integrity=False)

    st.write("""Conjunto de datos final:""")
    df
    excel_data = BytesIO()
    with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Hoja1')

    # Descargar el archivo Excel
    b64 = base64.b64encode(excel_data.getvalue()).decode('utf-8')
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="dataframe.xlsx">Descargar archivo Excel</a>'
    st.markdown(href, unsafe_allow_html=True)
    