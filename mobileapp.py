import pickle as pkl
pkf1=pkl.load(open('model2_mob.pkl','rb'))
print(pkf1)
import streamlit as st  
st.title("Mobile Price Prediction Model")
my=st.number_input("Enter Model Year")   
ram=st.number_input("Enter RAM in GB")  
s=st.number_input("Enter Storage in GB")
b=st.number_input("Enter Battery in mAh")
pc=st.number_input("Enter Primary Camera in MP")
if st.button("Predict"):
    result=pkf1.predict([[my,ram,s,b,pc]])
    st.success(f"The predicted price is {result[0]} USD")
    st.balloons()
    