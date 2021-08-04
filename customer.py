 
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import datetime
model = load_model('bank_model')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('bankoffice.jpg')
    
    
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict if a customer will deposit or not')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image)
    st.title("Depositor Or Not")
    
    if add_selectbox == 'Online':
        Age=st.number_input('Age:' , min_value=15, max_value=100, value=15)
        
        Job=st.selectbox("Job:", ('admin', 'blue-collar', 'entrepreneur','housemaid','management','retired','self-employed',
       'services','student','technician','unemployed','unknown'))
        Marital=st.selectbox("Marital", ('divorced', 'married', 'single'))
        Education=st.selectbox("Education", ('primary', 'secondary', 'tertiary','unknown'))
        Default = st.radio("Default", ('Yes', 'No'))
        Balance=st.number_input('Balance' , min_value=-300, max_value=30000, value=1)
        Housing = st.radio("Housing", ('Yes', 'No'))
        Loan = st.radio("Loan", ('Yes', 'No'))
        Contact=st.selectbox("Contact", ('cellular', 'telephone', 'unknown'))
        Day=st.number_input('Day' , min_value=1, max_value=31, value=1)
        Month=st.selectbox("Month", ('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'))
        Duration=st.number_input('Duration' , min_value=0, max_value=10000, value=1)
        Campaign=st.number_input('Campaign' , min_value=0, max_value=50, value=1)
        Pdays=st.number_input('Pdays' , min_value=-1, max_value=1000, value=1)
        Previous=st.number_input('Previous' , min_value=0, max_value=10, value=1)
        Poutcome=st.selectbox("Poutcome", ('failure', 'other', 'sucess','unknown'))
      
        
        output=" "
        input_dict={'Age':Age,'Job':Job,'Marital':Marital,'Education':Education,'Default': Default,'Balance':Balance,
        'Housing' : Housing,'Loan' : Loan,'Contact' : Contact,'Day' : Day,'Month' : Month,'Duration' : Duration,'Campaign' : Campaign,
        'Pdays' : Pdays,'Previous' : Previous,'Poutcome' : Poutcome}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('Depositor: {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
