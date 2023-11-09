# Importing Libraries
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Setting Webpage Configurations
st.set_page_config(page_title="Industrial Copper modeling", layout="wide")

eng = OrdinalEncoder()


# Reading the Dataframe
model_df = pd.read_csv('D:\Data_Excel\copper_dt.csv')


# Querying Win/Lost status
query_df = model_df.query("status == 'Won' or status == 'Lost'")

# model_df.drop(['item_date'],axis=1) 
# model_df.drop(['delivery date'],axis=1)

y = model_df['status']
x = model_df.drop(['status'], axis = 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)


model = DecisionTreeClassifier()
result = model.fit(x_train, y_train)

tab1,tab2 = st.tabs(['Selling Price Prediction','Status Prediction'])

with tab1:

    #item_year = st.selectbox('Select the Item year',options = query_df['item_date'].value_counts().index.sort_values())

    # item_year = st.number_input('Enter the date')
    # item_year=np.log(item_year)


    country = st.selectbox('Select the Country Code',options = query_df['country'].value_counts().index.sort_values())

    item_type = st.selectbox('Select the Item type',options = query_df['item type'].unique())

    application = st.selectbox('Select the Application number',options = query_df['application'].value_counts().index.sort_values())

    product_ref = st.selectbox('Select the Product Category',options = query_df['product_ref'].value_counts().index.sort_values())

    #delivery_year = st.selectbox('Select the Delivery year',options = query_df['delivery date'].value_counts().index.sort_values())

    # delivery_year = st.number_input('Enter the delivery date')
    # delivery_year=np.log(delivery_year)    

    thickness = st.number_input('Enter the Thickness')
    log_thickness = np.log(thickness)

    width = st.number_input('Enter the width')
    log_width = np.log(width)

    quantity_tons = st.number_input('Enter the Quantity (in tons)')
    log_quantity = np.log(quantity_tons)

    submit = st.button('Predict Price')

    if submit:
    
        user_input = pd.DataFrame([[country,item_type,application,log_thickness,log_width,product_ref,quantity_tons]],
                            columns = ['country','item type','application','thickness','width','product_ref','quantity_tons'])
        
        prediction = result.predict(user_input)
        
        selling_price = np.exp(prediction)
        st.subheader(f':green[Predicted Price] : {round(selling_price[0])}')

with tab2:

    country = st.selectbox('Select any one Country Code',options = query_df['country'].value_counts().index.sort_values())

    item_type = st.selectbox('Select any one Item type',options = query_df['item type'].unique())

    product_ref = st.selectbox('Select any one Product Category',options = query_df['product_ref'].value_counts().index.sort_values())

    #delivery_year = st.selectbox('Select a Delivery year',options = query_df['delivery date'].value_counts().index.sort_values())

    thickness = st.number_input('Enter an Thickness')
    log_thickness = np.log(thickness)

    width = st.number_input('Enter an width')
    log_width = np.log(width)

    selling_price = st.number_input('Enter an Selling Price')
    log_selling_price = np.log(selling_price)

    quantity_tons = st.number_input('Enter an Quantity (in tons)')
    log_quantity = np.log(quantity_tons)

    user_input_1 = pd.DataFrame([[country,item_type,log_thickness,log_width,product_ref,log_selling_price,log_quantity]],
                       columns = ['country','item type','thickness','width','product_ref','selling_price','quantity_tons'])
    
    submit1 = st.button('Predict')

    if submit1:
        transformed_data = eng.transform(user_input_1)
        prediction = result.predict(transformed_data)
        
        if prediction[0] == 1:
            st.subheader(':green[Predicted Status] : Won')
        else:
            st.subheader('green[Predicted Status] : Lost')