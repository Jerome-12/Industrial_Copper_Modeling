import pandas as pd
import numpy as np

df=pd.read_csv('D:\Data_Excel\Copper_Set.xlsx - Result 1.csv')

#print(df)

#print(df.head())

#print(df.shape)

#print(df.isnull().sum())
#print(df.info())

df['quantity tons']=pd.to_numeric(df['quantity tons'],errors='coerce')
df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date

df['material_ref'].fillna('unknown',inplace=True)

#print(df.info())
df.loc[df['material_ref'].str.startswith('000'), 'material_ref'] = np.nan

print(df.head())

#print(df.info())

print(df['status'].unique())

df1=df.dropna(axis=0)

# print(df1.info())

df1 = df1[df1['status'].isin(['Won', 'Lost'])]

print(df1['status'].unique())

print(df1.info())

df1 = df1.drop(['id','customer'],axis = 1)

print(df1.shape)

df1 = df1.drop(['item_date','delivery date'],axis = 1)


df1.to_csv('copper_dt.csv',index=False)


