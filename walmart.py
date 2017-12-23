from xgboost.sklearn import XGBClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import pickle

#import data
df = pd.read_csv('train.csv')

#Preprocessing and Data Munging
target_df = df[['TripType']]
#transform dataframe, or series, to a ndarray
target_vector = target_df.values
print(target_vector.shape)

categorical_df = df[['Weekday', 'DepartmentDescription']]
categorical_df = categorical_df.fillna('-1')
df['Upc'] = df.fillna(df['Upc'].mean())
df['FinelineNumber'] = df.fillna(df['FinelineNumber'].mean())
df = df.drop(['Weekday', 'DepartmentDescription'], axis=1)
df = df.fillna(0)
#Apply LabelEncoder
encoder = LabelEncoder()
temp_df = categorical_df.apply(encoder.fit_transform)

#combine data
df1 = df[['VisitNumber']].values
df2 = df[['Upc']].values
df3 = df[['FinelineNumber']].values

#prints = [print(x.shape) for x in [df1,df2,df3]]

print(df)
print(temp_df.shape)
final_train_data = np.concatenate((df1,df2,df3,temp_df), axis=1)

#train the model
model = XGBClassifier(silent=False)
print("Training the model....")
model.fit(final_train_data,target_vector)
print("Dumping model....")
f = open('walmart.pkl', 'wb+')
pickle.dump(model, f)
