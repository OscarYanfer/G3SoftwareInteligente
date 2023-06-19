import streamlit as st

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

import pandas as pd
from io import StringIO

st.markdown("# Sección I - Preprocesamiento")
st.write(
    """Ingrese un conjunto de datos."""
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

datos = uploaded_file

#Columna iniciasl será el index
df = pd.read_excel(datos, sheet_name='Sheet1', index_col = 0)
#df = pd.read_excel(datos, sheet_name='data')

df

#Eliminación de columnas con instancias únicas
df = df.drop(['CODIGO DE LA ENTIDAD', 'CODIGO UBIGEO INEI', 'CODIGO PAIS ', 'NOMBRE DE LA UO'], axis=1)
#df.drop(['ID', 'CODIGO DE LA ENTIDAD', 'CODIGO UBIGEO INEI', 'CODIGO PAIS ', 'NOMBRE DE LA UO'], axis=1)

#Verificación de valores vacíos
print(df.isnull().any().any())

#Filtración de valores pertenecientes al primer semestre del año 2021
df = df.loc[(df['Fecha_v2'] >= '2021-01-01')
                     & (df['Fecha_v2'] < '2021-07-01')]

#Visualización de la información
df.head()

df_temp = df
df_temp = df_temp.drop(['Fecha', 'Fecha_v2', 'Hora'], axis=1)

#Guardar en un array las columnas de tipo numérico
columnas = df_temp.columns

#Calculamos el promedio de cada columna numérica y la agregamos a las filas con información faltante e imprimimos lo que se cambiará
for c in columnas:
    mean = df[c].mean()
    print(mean)
    df[c] = df[c].fillna(mean)

#Verificación de valores vacíos
print(df.isnull().any().any())

df.head()

#Reemplazar valores en 0 con los valores máximos de cada columna 
for c in columnas:
    max = df[c].max()
    print(max)
    df[c] = df[c].replace({0: max})

df.head()

df = df.rename(columns={'Fecha':'Fecha y Hora', 'Fecha_v2':'Fecha'})
df.info()

#Transformamos la división de atributos en tipo fecha
df['Año'] = df['Fecha'].dt.year 
df['Mes'] = df['Fecha'].dt.month 
df['Dia'] = df['Fecha'].dt.day 

#Reordenamos las columnas
df = df[['Fecha y Hora', 'Fecha', 'Año', 'Mes', 'Dia', 'Hora', 'CO (ug/m3)', 'H2S (ug/m3)', 'NO2 (ug/m3)', 'O3 (ug/m3)',
       'PM10 \n(ug/m3)', 'PM2.5 \n(ug/m3)', 'SO2 (ug/m3)', 'Ruido (dB)', 'UV',
       'Humedad (%)', 'Latitud', 'Longitud', 'Presion \n(Pa)',
       'Temperatura (C)']]

df.head()

#Cambiando el indice a la Fecha y Hora, si es que se desea
df.set_index("Fecha y Hora", drop=True, append=False, inplace=True, verify_integrity=False)

df