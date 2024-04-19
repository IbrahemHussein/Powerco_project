import streamlit as st
import pandas as pd
import skops.io as sio
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import KFold, cross_val_score,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from yellowbrick.classifier import confusion_matrix as yell_con
from yellowbrick.classifier import ROCAUC
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler,MaxAbsScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')
import plotly_express as px


st.title('model page')
st.subheader('please select the features to input')


# Function to read uploaded CSV file
def read_uploaded_csv(uploaded_file):
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        return input_df
    return None

def discount_level(forecast_discount_energy):
    if forecast_discount_energy > 0 and forecast_discount_energy <=5 :
        discount_level='0-5'
    elif forecast_discount_energy > 5 and forecast_discount_energy <=10:
        discount_level='6-10'
    elif forecast_discount_energy > 10 and forecast_discount_energy <= 20:
        discount_level='11-20'
    elif forecast_discount_energy > 20 and forecast_discount_energy <=30:
        discount_level ='21-30'
    return discount_level

def user_input_features():
    col1, col2,col3 = st.columns(3)
    with col1:
            channel_sales = st.selectbox('Channel Sales', 
                                        ['foosdfpfkusacimwkcsosbicdxkicaua', 'MISSING',
                                        'lmkebamcaaclubfxadlmueccxoimlema', 'usilxuppasemubllopkaafesmlibmsdf',
                                        'ewpakwlliwisiwduibdlfmalxowmwpci'])
            cons_12m = st.number_input('Electricity Consumption Of The Past 12 Months', 0, 6207104, 300000)
            cons_gas_12m = st.number_input('Gas Consumption Of The Past 12 Months', 0, 4154590, 40000)
            cons_last_month = st.number_input('Electricity Consumption Of The Last Month', 0, 771203, 35000)
            forecast_price_energy_peak = st.slider("Forecasted Energy Price For 2nd Period (peak)", 0.0, 1.0, 0.025)
            imp_cons = st.slider("Current Paid Consumption", 0.0, 15050.79, 5000.0)
            offpeak_diff_dec_january_energy = st.slider("The Difference Between Off-Peak Energy Prices In December and January", -0.148477, 0.5, 0.0025)
            offpeak_diff_dec_january_power = st.slider("The Difference Between Off-Peak Power Prices In December and January.", -44.0, 50.0, 10.0)
            price_off_peak_var = st.slider("price of energy for the 1st period (off peak)", 0.0, 0.5, 0.254)
            price_peak_var = st.slider("price of energy for the 2nd period (peak)", 0.0, 0.5, 0.254)
            price_month = st.slider("The Month of Price", 0, 12, 2)
            modification_count = st.slider("Modification Count On Contract", 0, 15, 2)

    with col2:
            is_has_gas = st.selectbox('is  client also a gas',('True', 'False'))
            if is_has_gas=='True':
                has_gas='t'
            else:
                has_gas='f'
            forecast_cons_12m = st.number_input('Forecasted Electricity Consumption For Next 12 Months', 0, 771203, 35000)
            forecast_cons_year = st.number_input('Forecasted Electricity Consumption For The Next Calendar Year', 0, 771203, 35000)
            forecast_discount_energy = st.slider('Forecasted Value Of Current Discount', 0, 30, 15)
            forecast_meter_rent_12m = st.slider('Forecasted Bill Of Meter Rental For the Next 2 Months', 0, 650, 120)
            forecast_price_energy_off_peak = st.slider("Forecasted Energy Price For 1st Period (off peak)", 0.0, 1.0, 0.025)
            forecast_price_pow_off_peak = st.slider("Forecasted Power Price For 1st Period (off peak)", 0.0, 60.0, 24.0)
            day_activ = st.slider("Active data of client to Reference Date (2016, 1, 1)", 400, 5000, 1000)
            day_to_end = st.slider("day to end serves of client to Reference Date (2016, 1, 1)", 20, 600, 150)
            price_mid_peak_var = st.slider("price of energy for the 3rd period (mid peak)", 0.0, 0.5, 0.254)
            price_off_peak_fix = st.slider("price of power for the 1st period (off peak)", 0.0, 100.0, 30.0)

    with col3:
            origin_up = st.selectbox('Code Of The Electricity Campaign',('lxidpiddsbxsbosboudacockeimpuepw','kamkkxfxxuwbdslkwifmmcsiusiuosws',
                                                                        'ldkssxwpmemidmecebumciepifcamkci'))
            margin_gross_pow_ele = st.slider('Gross Margin On Power Subscription', 0.0, 400.0, 150.0)
            margin_net_pow_ele = st.slider('Net Margin On Power Subscription', 0.0, 400.0, 150.0)
            nb_prod_act = st.slider('Number Of Active Products And Services', 1, 50, 20)
            net_margin = st.slider('Total Net Margin', 1.0, 25000.0, 2000.0)
            num_years_antig = st.slider('Antiquity Of The Client (in number of years)', 0, 15, 6)
            pow_max = st.slider('subscribed power', 0.0, 400.0, 100.0)
            day_modif_prod = st.slider("day modification prod of client to Reference Date (2016, 1, 1)", 0, 5000, 2000)
            day_renewal = st.slider("day renewal of client to Reference Date (2016, 1, 1)", 0, 2000, 200)
            price_peak_fix = st.slider("price of power for the 2nd period (peak)", 0.0, 100.0, 30.0)
            price_mid_peak_fix = st.slider("price of power for the 3rd period (mid peak)", 0.0, 30.0, 11.0)



    # Create dictionary with input data
    data = {
        'channel_sales': channel_sales,
        'cons_12m': cons_12m,
        'cons_gas_12m': cons_gas_12m,
        'cons_last_month': cons_last_month,
        'forecast_cons_12m': forecast_cons_12m,
        'forecast_cons_year': forecast_cons_year,
        'forecast_discount_energy': forecast_discount_energy,
        'forecast_meter_rent_12m': forecast_meter_rent_12m,
        'forecast_price_energy_off_peak': forecast_price_energy_off_peak,
        'forecast_price_energy_peak': forecast_price_energy_peak,
        'forecast_price_pow_off_peak': forecast_price_pow_off_peak,
        'has_gas': has_gas,
        'imp_cons': imp_cons,
        'margin_gross_pow_ele': margin_gross_pow_ele,
        'margin_net_pow_ele': margin_net_pow_ele,
        'nb_prod_act': nb_prod_act,
        'net_margin': net_margin,
        'num_years_antig': num_years_antig,
        'origin_up': origin_up,
        'pow_max': pow_max,
        'descount_level': discount_level(forecast_discount_energy),
        'offpeak_diff_dec_january_energy': offpeak_diff_dec_january_energy,
        'offpeak_diff_dec_january_power': offpeak_diff_dec_january_power,
        'day_activ': day_activ,
        'day_to_end': day_to_end,
        'day_modif_prod': day_modif_prod,
        'day_renewal': day_renewal,
        'price_off_peak_var': price_off_peak_var,
        'price_peak_var': price_peak_var,
        'price_mid_peak_var': price_mid_peak_var,
        'price_peak_fix': price_peak_fix,
        'price_mid_peak_fix': price_mid_peak_fix,
        'price_off_peak_fix': price_off_peak_fix,
        'price_month': price_month,
        'modification_count': modification_count,
    }
    
    # Convert dictionary to DataFrame
    feature = pd.DataFrame(data, index=[0])
    return feature


uploaded_file = st.file_uploader('Update your CSV feature files', type=['csv'])
input_df = read_uploaded_csv(uploaded_file)
if input_df is not None:
    st.write('Uploaded DataFrame:')
    st.write(input_df)
else:
    st.write('No file uploaded yet.')

    # User input features
    st.subheader('User Input Features')
    input_df = user_input_features()
    st.write(input_df)

complte_pipline=sio.load('C:/Users/Ebrahim/Desktop/workshop/data scientist project/final project/PowerCo_project-20240416T151617Z-001/PowerCo_project/models/PowerCo_model.skops',trusted=True)

if st.button('Show Results', key='show_results_button',
            help='Click to show results',use_container_width=True):
    st.subheader('predictions')

    client_churn=np.array(['Not-Churn','Churn'])
    predictions=complte_pipline.predict(input_df)
    st.write(client_churn[predictions],)

    predictions_proba=complte_pipline.predict_proba(input_df)
    st.subheader('predictions_proba')
    st.write(predictions_proba)

    feature_names = complte_pipline.named_steps['preprocessing']
    feature_encoded=feature_names.get_feature_names_out()
    encoded_features = pd.Series(feature_encoded)
    encoded_features = encoded_features.str.replace('ohe__', '')
    encoded_features = encoded_features.str.replace('remainder__', '')
    encoded_features = encoded_features.str.replace('number__', '')
    encoded_features = encoded_features.str.replace('category__', '')
    importance = complte_pipline.named_steps['training_model'].named_steps['R_model'].feature_importances_

    feature_importances=pd.DataFrame({
        'Features':encoded_features,
        'Importances':np.round( importance*100,2)
    })
    feature_importances=feature_importances.sort_values(by='Importances')

    fig=px.bar(feature_importances,x='Importances',y='Features',text='Importances',
        title='The Importance of eatch Features',
        color_discrete_sequence=['orange'],width=1000,height=1000)

    fig.update_traces(texttemplate='%{text:.1f}',textposition='outside')
    fig.update_layout(title_font=dict(size=20),title_x=0.5,)
    fig.update_xaxes(title_font=dict(size=16),tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16),tickfont=dict(size=14))
    st.plotly_chart(fig)

st.markdown("""
    <style>
    .centered-button {
        display: block;
        margin: 0 auto;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)