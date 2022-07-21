import numpy as np
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.header("Prediction of C3S,C2S,C3A and C4AF")


df1=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/Jan14.csv', header=None)
df2=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/Feb14.csv', header=None)
df3=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/Mar14.csv', header=None)
df4=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/Apr14.csv', header=None)
df5=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/May14.csv', header=None)
df6=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/June14.csv', header=None)
df7=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/July14.csv', header=None)
df8=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/Sep14.csv', header=None)
df9=pd.read_csv('https://github.com/SaicharanRitwik39/HostMillProject/blob/main/Oct14.csv', header=None)

df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9])

df.dropna(inplace=True)
df=df[df['Al2O3']>1]
df.drop(['2C3A+C4AF'],axis=1,inplace=True)

df.rename(columns={'F.CaO':'FCaO','A/F':'AF','LIQUID %':'liquid','LT.WT Gm/Ltr':'LT_WT'},inplace=True)
st.write(df)
x=df.drop(['C3S','C2S','C3A','C4AF'],axis=1)
y=df[['C3S','C2S','C3A','C4AF']]

st.sidebar.header("Specify input parameters:")
def user_input():
    Al2O3=st.sidebar.slider('Al2O3',float(x.Al2O3.min()),float(x.Al2O3.max()),float(x.Al2O3.mean()))
    CaO=st.sidebar.slider('Cao',float(x.CaO.min()),float(x.CaO.max()),float(x.CaO.mean()))
    Fe2O3=st.sidebar.slider('Fe2O3',float(x.Fe2O3.min()),float(x.Fe2O3.max()),float(x.Fe2O3.mean()))
    MgO=st.sidebar.slider('MgO',float(x.MgO.min()),float(x.MgO.max()),float(x.MgO.mean()))
    SO3=st.sidebar.slider('SO3',float(x.SO3.min()),float(x.SO3.max()),float(x.SO3.mean()))
    SiO2=st.sidebar.slider('SiO2',float(x.SiO2.min()),float(x.SiO2.max()),float(x.SiO2.mean()))
    FCaO=st.sidebar.slider('F.CaO',float(x.FCaO.min()),float(x.FCaO.max()),float(x.FCaO.mean()))
    LSF=st.sidebar.slider('LSF',float(x.LSF.min()),float(x.LSF.max()),float(x.LSF.mean()))
    SM=st.sidebar.slider('SM',float(x.SM.min()),float(x.SM.max()),float(x.SM.mean()))
    AF=st.sidebar.slider('A/F',float(x.AF.min()),float(x.AF.max()),float(x.AF.mean()))
    LIQUID=st.sidebar.slider('LIQUID %',float(x.liquid.min()),float(x.liquid.max()),float(x.liquid.mean()))
    LT_WT=st.sidebar.slider('LT.WT Gm/Ltr',float(x.LT_WT.min()),float(x.LT_WT.max()),float(x.LT_WT.mean()))
    K2O=st.sidebar.slider('K2O',float(x.K2O.min()),float(x.K2O.max()),float(x.K2O.mean()))
    Na2O=st.sidebar.slider('Na2O',float(x.Na2O.min()),float(x.Na2O.max()),float(x.Na2O.mean()))
    TiO2=st.sidebar.slider('TiO2',float(x.TiO2.min()),float(x.TiO2.max()),float(x.TiO2.mean()))
    input={'Al2O3':Al2O3,
           'CaO':CaO,
           'Fe2O3':Fe2O3,
           'MgO':MgO,
           'SO3':SO3,
           'SiO2':SiO2,
           'FCaO':FCaO,
           'LSF':LSF,
           'SM':SM,
           'AF':AF,
           'LIQUID %':LIQUID,
           'LT.WT Gm/Ltr':LT_WT,
           'K2O':K2O,
           'Na2O':Na2O,
           'TiO2':TiO2}
    features=pd.DataFrame(input,index=[0])
    return features

input=user_input()

r=RandomForestRegressor(n_estimators=200)
r.fit(x,y)
prediction=r.predict(input)
pred=pd.DataFrame(data=prediction,columns=['C3S','C2S','C3A','C4AF'])
st.header("Predictions:")
st.write(pred)
