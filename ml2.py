
from enum import auto
from math import gamma
from optparse import Option
from select import select
from webbrowser import get
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC,SVR
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,mean_squared_error
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
       A simple ML data app with streamlit

""")

def EDA():
    st.subheader('Exploratory Data Analysis')
    data = st.file_uploader('Chosse your dataset',type=['csv','xlsx','txt','json'])
    
    if data is not None:
        df = pd.read_csv(data)
        st.success('Data is uploaded')
        st.dataframe(df.head(20))

        if st.checkbox('Shape'):
            st.write("The shape of your data:",df.shape)
        if st.checkbox('Columns Name'):
            st.write(df.columns)
        
        if st.checkbox("Check Null values"):
            st.write(df.isnull().sum())
        if st.checkbox('Select Columns'):
            selected_columns = st.multiselect("Columns are",df.columns)

            df1 = df[selected_columns]
            st.dataframe(df1)  
        if st.checkbox('Describe'):
            st.write(df1.describe().T)  
        if st.checkbox('Display Correlation of data variuos columns'):
            st.write(df.corr()) 
def viualization():
    c =0
    st.subheader('Exploratory Data Analysis')
    data = st.file_uploader('Chosse your dataset',type=['csv','xlsx','txt','json'])
    if data is not None:
        df = pd.read_csv(data)
        st.success('Data is uploaded')
        st.dataframe(df.head(20))
    
    if st.checkbox('countplot'):
        cselected_columns = st.selectbox('columns name',df.columns,key=c)
        c+=1
        plt.figure(figsize=(15,6))
        sns.countplot(df[cselected_columns])
        st.pyplot()
    if st.checkbox('Distplot'):
        dselected_columns = st.selectbox('columns name',df.columns,key=c)
        c+=1
        try :
            
            plt.figure(figsize=(15,6))
            sns.distplot(df[dselected_columns])
            st.pyplot()
        except:
            st.error('Plese select another column')
    if st.checkbox('Heatmap'):
        plt.figure(figsize=(15,6))
        sns.heatmap(df.corr(),annot=True)
        st.pyplot()
    if st.checkbox('Scatterplot'):
        sselected_columns = st.multiselect('columns name',df.columns,key=c)
        c+=1
        if len(sselected_columns) == 2:
            try :
                
                sns.scatterplot(df[sselected_columns[0]],df[sselected_columns[1]])
                st.pyplot()
            except:
                st.error('Plese select another column')
        else:
            st.error('Please select two columns')
    if st.checkbox('PairPlot'):
        plt.figure(figsize=(15,6))
        sns.pairplot(df,diag_kind='kde')
        st.pyplot()
    if st.checkbox('Boxplot'):
        bselected_columns = st.selectbox('columns name',df.columns,key=c)
        c+=1
        try :
           
            plt.figure(figsize=(15,6))
            sns.boxplot(df[bselected_columns])
            st.pyplot()
        except:
            st.error('Plese select another column')
    if st.checkbox('Histgram'):
        hselected_column = st.selectbox('columns name',df.columns,key=c)
        c+=1
        try :
            plt.figure(figsize=(15,6))
            sns.histplot(df[hselected_column])
            st.pyplot()
        except:
            st.error('Plese select another column')

    if st.checkbox('Piechart'):
        pieselected_column = st.selectbox('columns name',df.columns,key=c)
        fig = plt.gcf()
        fig.set_size_inches(4,4)
        c+=1
        try :
            
            piechart = df[pieselected_column].value_counts().plot.pie(autopct="%1.1f%%")
            st.write(piechart)
            st.pyplot()
        except:
            st.error('Plese select another column')

def add_param(name):
    param = {}
    if name == 'SVC' or name == 'SVR':
        c = st.sidebar.slider('C',0.01,15.0)
        gamma = st.sidebar.selectbox('gamma',('scale', 'auto'))
        kernal = st.sidebar.selectbox('Kernal',('linear', 'poly', 'rbf', 'sigmoid', ))
        param['C'] = c
        param['gamma'] = gamma
        param['kernel'] = kernal
    if name == 'KNeighborsClassifier' or name == 'KNeighborsRegressor':
        k = st.sidebar.slider('K',1,15)
        param['k'] = k
    
    return param
def get_model(name,param):
    if name == 'SVC':
        clf = SVC(C=param['C'],gamma=param['gamma'],kernel=param['kernel'])
    if name == 'KNeighborsClassifier':
        clf = KNeighborsClassifier(n_neighbors=param['k'])
    if name =='LogisticRegression':
        clf = LogisticRegression()
    if name == 'RandomForestClassifier':
        clf = RandomForestClassifier()
    if name == 'SVR':
        clf = SVR(C=param['C'],gamma=param['gamma'],kernel=param['kernel'])
    if name == 'KNeighborsRegressor':
        clf = KNeighborsRegressor(n_neighbors=param['k'])
    if name =='LinearRegression':
        clf = LinearRegression()
    if name == 'RandomForestRegressor':
        clf = RandomForestRegressor()
    return clf
def model():
    st.subheader('Exploratory Data Analysis')
    data = st.file_uploader('Chosse your dataset',type=['csv','xlsx','txt','json'])
    
    if data is not None:
        df = pd.read_csv(data)
        st.success('Data is uploaded')
        st.dataframe(df.head(20))

        if st.checkbox("select columns for Model Building"):
            st.write("Note : Target columns should be in the last")
            column = st.multiselect('Select',df.columns)
            df1 = df[column]
            st.dataframe(df1)
            x = df1.iloc[:,:-1]
            y = df.iloc[:,-1]
            seed = st.sidebar.slider('Chosse the seed',1,200)

            
            # getting algorithum
            
        
        clf_list =['RandomForestClassifier','LogisticRegression','KNeighborsClassifier','SVC','RandomForestRegressor','LinearRegression','KNeighborsRegressor','SVR']
        clf = st.sidebar.selectbox('Select a algorithum',clf_list)
        param = add_param(clf)
        model = get_model(clf,param)
        if st.checkbox('Build'):
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
            model.fit(X_train,y_train)

            pred = model.predict(X_test)
            try:
                score = accuracy_score(y_test,pred)
                st.write("Current Classifier Name :",model)
                st.write("accuracy score :",score)
            except:
                score = np.sqrt(mean_squared_error(y_test,pred))
                
                st.write("Current Regression Name :",model)
                st.write("Root Mean Squared Error :",round(score,2))

def main():
    activities = ['EDA','visualization','Model','About us']
    option = st.sidebar.selectbox('Select a option',activities)
    if option == 'EDA':
        EDA()
        
    if option == 'visualization':
        viualization()
    if option == 'Model':
        
        model()
    if option == 'About us':
        st.subheader("Made By Biltu Dey")
        st.write('[Github](https://github.com/biltudey)')
        st.write('My Email : biltudey222@gmail.com')
        st.write('[Linkdin](https://www.linkedin.com/in/biltudey)')
        st.balloons()
        st.snow()

if __name__ == '__main__':
    main()