import streamlit as st
import pandas as pd

@st.cache_data
def get_data():
    df = pd.read_csv("Dataset/dairy_dataset.csv")
    df.rename(columns={"Product Name": "Product_Name","Customer Location":"Customer_Location"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month_name()
    df["Year"] = df["Date"].dt.year
    return df
