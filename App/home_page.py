import streamlit as st
import pandas as pd



st.set_page_config(
    page_title='PowerCo Streamlit App',
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('PowerCo App')

st.info('PowerCo is a major gas and electricity utility that supplies to corporate, SME (Small & Medium Enterprise), and residential customers.')
st.image('app_image.jpeg', caption='Project image',width=500)

st.subheader('Project Overview')
st.write('This project involves analyzing PowerCo\'s customer data to understand the source of customer churn. The data includes various aspects of PowerCo\'s operations and customer behavior, providing a comprehensive view of the company\'s situation.')


st.subheader('Data Description')

st.write('The data includes information about PowerCo\'s customers and their consumption patterns, contract details, forecasted consumption and prices, and churn status. It also includes price data for different periods.')
df=pd.read_csv('../data/processed/client_clean_plus.csv')
subset_data = df.head(5)
st.dataframe(subset_data)


