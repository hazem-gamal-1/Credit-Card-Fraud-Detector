import pandas as pd
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import  RandomUnderSampler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from imblearn.pipeline import Pipeline
import numpy as np
def Load_data(path):
    df=pd.read_csv(path)
    df['Amount']=np.log1p(df['Amount'])
    df=df.to_numpy()
    X=df[:,:-1]
    T=df[:,-1]
    return  X,T


def Get_Sampler(option, ratio,size):
    size_of_postive=int(ratio*size)
    size_of_negative=int((1-ratio)*size)
    Sampler = {
        1: Pipeline(steps=[('sampler', RandomOverSampler(sampling_strategy={1: size_of_postive}, random_state=42))]),
        2: Pipeline(steps=[('sampler', RandomUnderSampler(sampling_strategy={0: size_of_negative}, random_state=42))]),
        3: Pipeline(steps=[('sampler', SMOTE(sampling_strategy={1: size_of_postive}, k_neighbors=7, random_state=42))]),
        4: Pipeline(steps=[
            ('under_sampler', RandomUnderSampler(sampling_strategy={0: size_of_negative}, random_state=42)),
            ('over_sampler', RandomOverSampler(sampling_strategy={1: size_of_postive}, random_state=42))
        ]),
        5: Pipeline(steps=[
            ('under_sampler', RandomUnderSampler(sampling_strategy={0: size_of_negative}, random_state=42)),
            ('smote', SMOTE(sampling_strategy={1: size_of_postive}, k_neighbors=7, random_state=42))
        ]),
    }
    return Sampler[option]
def Get_preprocessor(option):
   preprocessor={
       1: MinMaxScaler(),
       2: StandardScaler(),
   }
   return preprocessor[option]





