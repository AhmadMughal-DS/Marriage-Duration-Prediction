import streamlit as st
import pandas as pd
import pickle
import joblib
#---------------Here import model
model = joblib.load('pipe.joblib')

df = pd.read_csv("divorce.csv",usecols=['education_man','income_man','education_woman','income_woman','num_kids','marriage_duration'])
st.title("Marriage Duration Prediction")


#-------------------------------Here is expander
head = df.head(5)
with st.expander("Here is dataset"):
    st.write(head)
st.caption("""**Marriage_Duration is our target input** \n
education_man,education_woman,income_man,income_woman,num_kids is input variable""")


#------------------------------Here is side bar
with st.sidebar:
    st.write("evething is good")


#-------------------------------Here is inputs
col1,col2,col3 = st.columns(3)
with col1:
    education_man = st.selectbox('education_man',df['education_man'].unique())
    income_man = st.number_input("income_man",min_value=0)
with col2:
    education_woman = st.selectbox('education_woman',df['education_woman'].unique())
    income_woman = st.number_input("income_woman",min_value=0)
with col3:
    num_kids = st.number_input("num_kids",min_value=0)

def get_user_input():
    input_data = {
        'education_man': [education_man],  # Note the extra brackets
        'income_man': [income_man],  # Note the extra brackets
        'education_woman': [education_woman],  # Note the extra brackets
        'income_woman': [income_woman],  # Note the extra brackets
        'num_kids': [num_kids]  # Note the extra brackets
    }
    user_input =  pd.DataFrame(input_data)
    return user_input
st.subheader("Step 2: Ask the model for a prediction")
#-------------------------------Here is prediction
primary_button =  st.button("Predict",type='primary')
if primary_button:
    user_input = get_user_input()
    answer = model.predict(user_input)
    st.write(answer)
