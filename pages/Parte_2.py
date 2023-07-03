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
#Sección III
st.markdown("# Sección II - Reducción de variables")
st.write("""NOTA: Ingresar archivo descargado en la sección I""")
uploaded_file = st.file_uploader("Cargar archivo Excel:", type=["xlsx"])
if uploaded_file is not None:
    #Columna iniciasl será el index
    st.write("""Conjunto de datos cargado: """)
    df = pd.read_excel(uploaded_file, sheet_name='Hoja1', index_col=0)
    
    #Convertir en 0 y 1 los valores de PM2.5, Niveles menores a 15 se consideran buenos, mientras que mayores se consideran perjudicial para el ser humano
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
    #Las variables no numéricas se transformaran a tipo objeto
    vf_string=X.columns[X.dtypes=="object"]
    df_string=X.loc[:,vf_string]
    #Verificamos la existencia de valores perdidos
    st.write("""Verificación de la existencia de valores perdidos:""")
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

    # Importamos la data procesada
    data = df
    ## Revisamos los datos nulos
    heatmap = sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    
    plt.title('Mapa de calor de valores nulos')
    st.pyplot(heatmap.figure)

    # Eliminamos los valores nulos si en el caso de que el conjunto de datos tenga pero no haya valores nulos que sean buenos
    data=data.dropna()
    
    save=data["PM2.5 \n(ug/m3)"].copy()
    data=data.drop("PM2.5 \n(ug/m3)",axis=1)
    data["PM2.5 \n(ug/m3)"]=save
    data=data.drop(['Año', 'Mes', 'Dia', 'Hora'], axis=1)

    # Asignamos los atirbutos dependientes e independientes
    X=data.iloc[:,:-1] # Atributos independientes
    y=data.iloc[:,-1] # Atributos dependientes

    ## Revisamos los valores nulos
    X.isnull()
    st.markdown("## Matriz de Correlación con Mapa de Calor")
    st.write("##### La correlación indica cómo se relacionan las características entre sí o con la variable de destino. La correlación puede ser positiva (el aumento de un valor de característica aumenta el valor de la variable objetivo) o negativa (el aumento de un valor de característica disminuye el valor de la variable objetivo). El mapa de calor facilita la identificación de qué características están más relacionadas con la variable objetivo, trazaremos un mapa de calor de las características correlacionadas utilizando la biblioteca seaborn.")

    import seaborn as sns
    # Obtener correlaciones de cada entidad en el conjunto de datos
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    # Plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    st.pyplot(g.figure)

    st.markdown("## Importancia de los atributos")
    st.write("##### Puede obtener la importancia de los atributos de cada entidad del conjunto de datos se usa la propiedad model.feature_importances_ del modelo. La importancia de la característica le da una puntuación para cada columna de los datos, cuanto mayor sea la puntuación, más importante o relevante será la característica para su variable de salida. La importancia de la característica es una clase incorporada que viene con un Regresor Basado en árbol, usaremos Extratreesregressor para extraer las 3 características principales para el conjunto de datos.")
    
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

    st.markdown("## Entrenamiento del modelo (Random Forest)")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor()
    regressor.fit(X_train,y_train)
    from sklearn.model_selection import cross_val_score
    score=cross_val_score(regressor,X,y,cv=5)



    st.markdown("## Métricas Obtenidas")
    #Aplicacion de hiperparametros
    RandomForestRegressor()
    from sklearn.model_selection import RandomizedSearchCV
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
    # Randorizamos la búsqueda de la data
    # Número de árboles en el random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
    # Número de atributos a considerar en cada split
    max_features = ['auto', 'sqrt']
    # Máximo de número de árboles
    max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
    # max_depth.append(None)
    # Mínimo de número de ejemplares requerido para separar un nodo
    min_samples_split = [2, 5, 10, 15, 100]
    # Número mínimo de muestras requeridas en cada nodo de hoja
    min_samples_leaf = [1, 2, 5, 10]
    # Creamos un grid aleatorio
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}
    # Usaremos el grid aleatorio para buscar los mejores hiperparámetros
    # Primero se crea el modelo base para afinar
    st.write("""Espere un momento...""")
    rf = RandomForestRegressor()
    # Búsqueda aleatoria de parámetros, mediante validación cruzada 3 veces,
    # Busca en 100 combinaciones diferentes
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter= 10, cv=5, verbose=2, random_state=42, n_jobs = 1)
    rf_random.fit(X_train,y_train)
    #rf_random.best_params_
    #rf_random.best_score_
    predictions=rf_random.predict(X_test)
    from sklearn import metrics
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Muestra las métricas en Streamlit
    st.write('MAE:', mae)
    st.write('MSE:', mse)
    st.write('RMSE:', rmse)

    #Creamos una función para predecir
    def predict_(CO, H2S, PM25, O3, PM10, SO2, R, UV, H, L, Lg, P, T):
        co = int(CO)
        h2 = float(H2S)
        pm25 = float(PM25)
        o3 = float(O3)
        p10 = int(PM10)
        so = float(SO2)
        r = float(R)
        uv = float(UV)
        h = float(H)
        l = float(L)
        lg = float(Lg)
        p = float(P)
        t = float(T)

        x = [[co,h2,pm25,o3,p10,so,r,uv,h,l,lg,p,t]]

        return rf_random.predict(x)
    
    # Ingresamos una secuencia de data, según lo determinado en la función predict_
    predictions = predict_(150.2, 15.2, 50.24, 48.15, 55.2, 11.4, 89.8, 4.8, 117.6, -13.0, -70.5, 1048.3, 19.4)[0]
    if predictions:
        st.write("Valores ingresados para comprobar el modelo:")
        st.write("CO: 150.2")
        st.write("H2S: 15.2")
        st.write("NO2: 50.24")
        st.write("O3: 48.15")
        st.write("PM10: 55.2")
        st.write("SO2: 11.4")
        st.write("Ruido: 89.8")
        st.write("UV: 4.8")
        st.write("Humedad: 117.6")
        st.write("Latitud: -13.0")
        st.write("Longitud: -70.5")
        st.write("Presión: 1048.3")
        st.write("Temperatura: 19.4")
        st.write('Valor de PM25 2.5 es:', predictions*100)
        color = ''
        if 0 <= predictions*100 <= 50:
            color = 'green'
        elif 51 <= predictions*100 <= 100:
            color = 'yellow'
        elif 101 <= predictions*100 <= 150:
            color = 'orange'
        elif 151 <= predictions*100 <= 200:
            color = 'red'
        elif 201 <= predictions*100 <= 300:
            color = 'purple'
        else:
            color = 'brown'

        # Mostrar la calidad del aire y el color correspondiente
        st.markdown(f'<span style="color: {color}; font-weight: bold;">'
                    f'La calidad del aire es {color.capitalize()}</span>',
                    unsafe_allow_html=True)
        
        st.title("Índice de Calidad del Aire (ICA)")
        st.write("Buena: Color verde (ICA de 0 a 50)")
        st.write("Moderada: Color amarillo (ICA de 51 a 100)")
        st.write("Dañina a la salud para grupos sensibles: Color naranja (ICA de 101 a 150)")
        st.write("Dañina a la salud: Color rojo (ICA de 151 a 200)")
        st.write("Muy dañina a la salud: Color morado (ICA de 201 a 300)")
        st.write("Peligrosa: Color marrón (ICA superior a 300)")
        

    
