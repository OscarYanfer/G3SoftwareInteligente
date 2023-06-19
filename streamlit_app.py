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
    st.write("""Conjunto de datos cargado:""")
    df = pd.read_excel(uploaded_file, sheet_name='Sheet1', index_col=0)
    st.dataframe(df)  # Mostrar el DataFrame en Streamlit

#Eliminación de columnas con instancias únicas
df = df.drop(['CODIGO DE LA ENTIDAD', 'CODIGO UBIGEO INEI', 'CODIGO PAIS ', 'NOMBRE DE LA UO'], axis=1)
#df.drop(['ID', 'CODIGO DE LA ENTIDAD', 'CODIGO UBIGEO INEI', 'CODIGO PAIS ', 'NOMBRE DE LA UO'], axis=1)

#Verificación de valores vacíos
st.write("""Conjunto de datos cargado: """+df.isnull().any().any())
#print(df.isnull().any().any())

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