#Importación de librerías
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
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

#Título de la página
st.set_page_config(
    page_title="Proyecto de Software Inteligente",
    page_icon="🤖",
)

#Encabezados principales
st.write("# Predicción de la calidad del aire en Miraflores - Lima")
st.markdown(
    """
    ## Grupo 3 - Integrantes:
    | Nombre | Participación|
    |--|--|
    | Oscar Stalyn Yanfer Laura | 19200260 |
    | Diego Tharlez Montalvo Ortega | 19200088 |
    | Jorge Luis Quispe Alarcon | 19200XXX |
    | Wilker Edison Atalaya Ramirez | 19200XXX |
    | Anthony Elias Ricse Perez | 19200XXX |

    ## Especificaciones:
    **Proyecto desplegado desde Streamlit**
    - Inclusión de preprocesamiento. 
    - Inclusión de gráficos
    - Inclusión de predicción
    """
)

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

    #Sección 2
    st.markdown("# Sección II - Exploración de modelos")
    
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

    #Sección III
    st.markdown("# Sección III - Filtro ANOVA (Reducción de atributos)")
    #Convertir en 0 y 1 los valores de PM2.5, Niveles menores a 15 se consideran buenos, mientras que mayores se consideran perjudicial para el ser humano
    df
    df['PM2.5 \n(ug/m3)']=df['PM2.5 \n(ug/m3)'].astype(float)
    df["PM2.5 \n(ug/m3)"]=np.where(df['PM2.5 \n(ug/m3)']<15, 0, 1)
    st.write("""NOTA: Convertir en 0 y 1 los valores de PM2.5, Niveles menores a 15 se consideran buenos, mientras que mayores se consideran perjudicial para el ser humano""")
    with st.spinner("Realizando conversión..."):
        time.sleep(1)
    st.write("""Conversión realizada""")
    #División de variable objetivo
    X=df.drop(['PM2.5 \n(ug/m3)'], axis=1)
    Y=df["PM2.5 \n(ug/m3)"]
    #Transformamos los atributos sobrantes a float64
    vf_float=X.columns[X.dtypes=="float64"]
    df_float=X.loc[:,vf_float]
    df_float
    #Las variables no numéricas se transformaran a tipo objeto
    vf_string=X.columns[X.dtypes=="object"]
    df_string=X.loc[:,vf_string]
    df_string
    #Verificamos la existencia de valores perdidos
    import missingno as msno 
    msno.bar(df_float)
    st.pyplot()
