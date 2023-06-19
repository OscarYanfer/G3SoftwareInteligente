import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Proyecto de Software Inteligente",
    page_icon="🤖",
)

st.write("# Predicción de la calidad del aire en Miraflores - Lima")
st.markdown(
    """
    ## Grupo G - Integrantes:
    | Nombre | Participación|
    |--|--|
    | Oscar Stalyn, Yanfer Laura | 19200260 |
    | Diego Tharlez Montalvo Ortega | 19200088 |
    | Jorge Luis Quispe Alarcon | 19200XXX |
    | Wilker Edison,Atalaya Ramirez | 19200XXX |
    | Anthony Elias,Ricse Perez | 19200XXX |

    ## Especificaciones:
    **Proyecto desplegado desde Streamlit**
    - Inclusión de preprocesamiento. 
    - Inclusión de gráficos
    - Inclusión de predicción
    """
)

st.markdown("# Sección I - Preprocesamiento")

uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    #Columna iniciasl será el index
    st.write("""Conjunto de datos cargado: """)
    df = pd.read_excel(uploaded_file, sheet_name='Sheet1', index_col=0)
    st.dataframe(df)  # Mostrar el DataFrame en Streamlit
    df = df.drop(['CODIGO DE LA ENTIDAD', 'CODIGO UBIGEO INEI', 'CODIGO PAIS ', 'NOMBRE DE LA UO'], axis=1)

#Eliminación de columnas con instancias únicas
#df = df.drop(['CODIGO DE LA ENTIDAD', 'CODIGO UBIGEO INEI', 'CODIGO PAIS ', 'NOMBRE DE LA UO'], axis=1)
#df.drop(['ID', 'CODIGO DE LA ENTIDAD', 'CODIGO UBIGEO INEI', 'CODIGO PAIS ', 'NOMBRE DE LA UO'], axis=1)

#Verificación de valores vacíos
st.write("Existencia de datos vacíos: " + str(df.isnull().any().any()))
#print(df.isnull().any().any())

st.write("""Filtración de valores pertenecientes al primer semestre del año 2021...""")
#Filtración de valores pertenecientes al primer semestre del año 2021
df = df.loc[(df['Fecha_v2'] >= '2021-01-01')
                     & (df['Fecha_v2'] < '2021-07-01')]

#Visualización de la información
#df.head()

df_temp = df
df_temp = df_temp.drop(['Fecha', 'Fecha_v2', 'Hora'], axis=1)

#Guardar en un array las columnas de tipo numérico
columnas = df_temp.columns

st.write("""Calculando el promedio de cada columna numérica...""")
st.write("""Agregando las filas con información faltante...""")
#Calculamos el promedio de cada columna numérica y la agregamos a las filas con información faltante e imprimimos lo que se cambiará
for c in columnas:
    mean = df[c].mean()
    print(mean)
    df[c] = df[c].fillna(mean)

#Verificación de valores vacíos
st.write("Existencia de datos vacíos: " + str(df.isnull().any().any()))

#df.head()

st.write("""Reemplazando valores en 0 con los valores máximos de cada columna...""")
#Reemplazar valores en 0 con los valores máximos de cada columna 
for c in columnas:
    max = df[c].max()
    print(max)
    df[c] = df[c].replace({0: max})

#df.head()

df = df.rename(columns={'Fecha':'Fecha y Hora', 'Fecha_v2':'Fecha'})
#df.info()

#Transformamos la división de atributos en tipo fecha
df['Año'] = df['Fecha'].dt.year 
df['Mes'] = df['Fecha'].dt.month 
df['Dia'] = df['Fecha'].dt.day 

st.write("""Reordenando columnas...""")
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

#######Librerías#######
import pandas as pd
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
seed = 4353

data = df
data.columns = data.columns.str.lower()
data.isnull().sum()
data.dropna(how='all',inplace=True)
data.dropna(thresh=10,axis=0,inplace=True)
data=data.drop([ 'año', 'mes', 'dia', 'hora'],axis=1)
st.write("""Aplicando modelos de Random Forest, Decision Tree Regression, Support Vector Machine, Regresión Lineal:""")
###Modelo de Regresión Lineal
from sklearn.preprocessing import StandardScaler         
from sklearn.model_selection import train_test_split      
from sklearn.linear_model import LinearRegression         
from sklearn.metrics import mean_squared_error,mean_absolute_error
col_=data.columns.tolist()[2:]
X=data[col_].drop('pm2.5 \n(ug/m3)',axis=1) 
y=data['pm2.5 \n(ug/m3)'] 
X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=42)
lr=LinearRegression()
lr_model=lr.fit(X_train,y_train)
y_pred=lr_model.predict(X_test)                      
rmse=np.sqrt(mean_squared_error(y_test,y_pred))      
###Modelo de Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor  
rf_reg=RandomForestRegressor()
rf_model=rf_reg.fit(X_train,y_train)           
y_pred_rf=rf_model.predict(X_test)  
rmseRF=np.sqrt(mean_squared_error(y_test,y_pred_rf))
###Modelo de Random Forest
from sklearn.tree import DecisionTreeRegressor         
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import GridSearchCV        
dt_one_reg=DecisionTreeRegressor()
dt_model=dt_one_reg.fit(X_train,y_train)         
y_pred_dtone=dt_model.predict(X_test)            
rmseMt=np.sqrt(mean_squared_error(y_pred_dtone,y_test))
###Modelo de Support Vector Machine
from sklearn.svm import SVR          
sv_reg=SVR()
sv_model=sv_reg.fit(X_train,y_train)
y_pred_sv=sv_model.predict(X_test) 
rmseSVM=np.sqrt(mean_squared_error(y_test,y_pred_sv))
###Comparando los modelos según el RMSE
st.write("""Comparando los modelos según el RMSE...""")
model = ['MLR', 'Random Forest', 'Tree Regression', 'SVM']
acc = [rmse, rmseRF, rmseMt, rmseSVM]

sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel("RMSE")
plt.xlabel("Modelos de Machine Learning")
sns.barplot(x=model, y=acc)
plt.xticks(rotation=45)
st.pyplot(plt)