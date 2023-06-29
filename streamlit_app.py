#Importación de librerías
import streamlit as st
warnings.filterwarnings("ignore")

#Título de la página
st.set_page_config(
    page_title="Proyecto de Software Inteligente",
    page_icon="🤖",
)
st.sidebar.success("Seleccione un modelo del menú")
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
