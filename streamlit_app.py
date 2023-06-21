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
import missingno as msno 
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
    st.markdown("# Sección III - Reducción de variables")
    #Convertir en 0 y 1 los valores de PM2.5, Niveles menores a 15 se consideran buenos, mientras que mayores se consideran perjudicial para el ser humano
    df['pm2.5 \n(ug/m3)']=df['pm2.5 \n(ug/m3)'].astype(float)
    df["pm2.5 \n(ug/m3)"]=np.where(df['pm2.5 \n(ug/m3)']<15, 0, 1)
    st.write("""NOTA: Convertir en 0 y 1 los valores de PM2.5, Niveles menores a 15 se consideran buenos, mientras que mayores se consideran perjudicial para el ser humano""")
    with st.spinner("Realizando conversión..."):
        time.sleep(1)
    st.write("""Conversión realizada""")
    #División de variable objetivo
    X=df.drop(['pm2.5 \n(ug/m3)'], axis=1)
    Y=df["pm2.5 \n(ug/m3)"]
    #Transformamos los atributos sobrantes a float64
    vf_float=X.columns[X.dtypes=="float64"]
    df_float=X.loc[:,vf_float]
    #Las variables no numéricas se transformaran a tipo objeto
    vf_string=X.columns[X.dtypes=="object"]
    df_string=X.loc[:,vf_string]
    #Verificamos la existencia de valores perdidos
    st.write("""Verificación de la existencia de valores perdidos""")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    msno.bar(df_float)
    st.pyplot()
    msno.bar(df_string)
    st.pyplot()
        
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from matplotlib import pyplot
    # Preparacion
    def prepare_inputs(X):
        oe = OrdinalEncoder()
        oe.fit(X)
        X_enc = oe.transform(X)
        return X_enc

    # feature selection
    def select_features_chi2(X, Y):
        fs = SelectKBest(score_func=chi2, k='all')
        fs.fit(X, Y)
        X_fs = fs.transform(X)
        return X_fs,fs
    #Preparamos la entradas
    X_train_enc= prepare_inputs(df_string)

    # feature selection
    _fs,fs = select_features_chi2(X_train_enc,Y)
    # what are scores for the features
    feature=[]
    for i in range(len(fs.scores_)):
        #print('Feature %s: %f' % (df_cat_tr.columns[i], fs.scores_[i]))
        feature.append([df_string.columns[i],fs.scores_[i]])
    df_feature = pd.DataFrame(feature, columns = ['Variable','Score'])

    df_feature=df_feature.sort_values('Score',ascending=False).reset_index(drop=True)
    #df_feature.iloc[0:50].to_excel("var_imp_cat.xlsx")

    df_feature1=df_feature.iloc[0:10].sort_values('Score',ascending=True).reset_index(drop=True)
    Peso = df_feature1['Score'].to_numpy()
    Variable = df_feature1['Variable'].to_numpy()

    importCHI,ax=plt.subplots(figsize=(10,10))
    st.markdown("## Chi cuadrado - Variabe más importante (no numérico)")
    plt.yticks(fontsize= 13)
    plt.xticks(fontsize= 12)
    plt.ylabel("Variables", fontsize=25)
    plt.xlabel("Score", fontsize=25)
    plt.barh(Variable,Peso)
    st.pyplot() 
    
    st.markdown("## Filtro ANOVA - Variabe más importante (numérico)")
    from sklearn import preprocessing
    df_float_z_score=pd.DataFrame(preprocessing.scale(df_float))
    # feature selection
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    from matplotlib import pyplot
    def select_features_anova(X, y):
        # configure to select all features
        fs = SelectKBest(score_func=f_classif, k='all')
        # learn relationship from training data
        fs.fit(X, y)
        # transform train input data
        X_train_fs = fs.transform(X)
        # transform test input data
        X_test_fs = fs.transform(X)
        return X_train_fs, fs
    # feature selection
    X_fs, fs = select_features_anova(df_float_z_score, Y)

    feature=[]
    for i in range(len(fs.scores_)):
        #print('Feature %s: %f' % (df_numericas_limp_tr.columns[i], fs.scores_[i]))
        feature.append([df_float.columns[i],fs.scores_[i]])
    df_feature_num = pd.DataFrame(feature, columns = ['Variable','Score'])

    df_feature_num=df_feature_num.sort_values('Score',ascending=False).reset_index(drop=True)
    #Score de variables importantes mediante filtro anova (numéricos)
    df_feature1=df_feature_num.iloc[0:13].sort_values('Score',ascending=True).reset_index(drop=True)
    Peso = df_feature1['Score'].to_numpy()
    Variable = df_feature1['Variable'].to_numpy()

    importCHI,ax=plt.subplots(figsize=(10,10))

    plt.yticks(fontsize= 13)
    plt.xticks(fontsize= 12)
    plt.ylabel("Variables", fontsize=25)
    plt.xlabel("Score", fontsize=25)
    plt.barh(Variable,Peso)
    st.pyplot() 
    
    #Sección 2
    st.markdown("# Sección IV - Aplicación de Random Forest")
    #Importamos las librerías
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ignoramos los avisos de warning
    import warnings
    warnings.filterwarnings('ignore')

    # Importamos la data procesada
    data = df
    ## Revisamos los datos nulos
    heatmap = sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Mapa de calor de valores nulos')
    st.pyplot(heatmap.figure)

    # Eliminamos los valores nulos si en el caso de que el conjunto de datos tenga pero no haya valores nulos que sean buenos
    data=data.dropna()
    
    save=data["pm2.5 \n(ug/m3)"].copy()
    data=data.drop("pm2.5 \n(ug/m3)",axis=1)
    data["pm2.5 \n(ug/m3)"]=save

    data=data.drop(['fecha', 'año', 'mes', 'dia', 'hora'], axis=1)

    # Asignamos los atirbutos dependientes e independientes
    X=data.iloc[:,:-1] # Atributos independientes
    y=data.iloc[:,-1] # Atributos dependientes

    ## Revisamos los valores nulos
    X.isnull()
    st.markdown("#### Matriz de Correlación con Mapa de Calor")
    st.markdown("##### La correlación indica cómo se relacionan las características entre sí o con la variable de destino.")
    st.markdown("##### La correlación puede ser positiva (el aumento de un valor de característica aumenta el valor de la variable objetivo) o negativa (el aumento de un valor de característica disminuye el valor de la variable objetivo).")
    st.markdown("##### El mapa de calor facilita la identificación de qué características están más relacionadas con la variable objetivo, trazaremos un mapa de calor de las características correlacionadas utilizando la biblioteca seaborn.")

    import seaborn as sns
    # Obtener correlaciones de cada entidad en el conjunto de datos
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    # Plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    st.pyplot(g.figure)

    st.markdown("#### Importancia de los atributos")
    st.markdown("##### Puede obtener la importancia de los atributos de cada entidad del conjunto de datos se usa la propiedad model.feature_importances_ del modelo.")
    st.markdown("##### La importancia de la característica le da una puntuación para cada columna de los datos, cuanto mayor sea la puntuación, más importante o relevante será la característica para su variable de salida.")
    st.markdown("##### La importancia de la característica es una clase incorporada que viene con un Regresor Basado en árbol, usaremos Extratreesregressor para extraer las 10 características principales para el conjunto de datos.")
    
    from sklearn.ensemble import ExtraTreesRegressor
    import matplotlib.pyplot as plt
    model = ExtraTreesRegressor()
    model.fit(X,y)

    #Sustentación de los valores obtenidos por el FILTRO ANOVA.
    # Gráfico del plot de la importancia de los atributos para una mejor visualización, el cual coincide con el filtro anova
    # Genera los datos de ejemplo
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    # Obtén los 5 valores más grandes y crea el gráfico de barras horizontal
    top_features = feat_importances.nlargest(3)
    fig, ax = plt.subplots()
    top_features.plot(kind='barh')
    plt.title('Importancia de características')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    # Muestra el gráfico en Streamlit
    st.pyplot(fig)

    st.markdown("#### Train Test split")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor()
    regressor.fit(X_train,y_train)
    from sklearn.model_selection import cross_val_score
    score=cross_val_score(regressor,X,y,cv=5)
    st.markdown("#### Evaluación del modelo")
    prediction=regressor.predict(X_test)
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, prediction)
    ax.set_title('Gráfico de dispersión entre y_test y prediction')
    ax.set_xlabel('y_test')
    ax.set_ylabel('prediction')

    # Establece el rango en 100 para ambos ejes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Muestra el gráfico en Streamlit
    st.pyplot(fig)  