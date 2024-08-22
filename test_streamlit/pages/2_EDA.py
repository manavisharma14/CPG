import pandas as pd
import streamlit as st
import seaborn as sns
import io
from scripts.data import get_data

st.set_page_config(page_title="EDA", page_icon="📊",layout="wide",)



st.markdown("# EDA")

st.write(
    """Exploratory Data Analysis"""
)

def make_layout(df):
    with st.container():
        st.header("Showcasing the dataset")
        st.dataframe(df)

    st.markdown("<h1 style='text-align: center; color: red;'>Basics of EDA</h1>", unsafe_allow_html=True)

    # ** head,tail,sample,columns,info,dtype
    with st.container():
        c1,c2 = st.columns(2)
        with c1:
            c1.header("Head")
            st.dataframe(df.head())
        with c2:
            c2.header("Tail")
            st.dataframe(df.tail())

    st.markdown("***")

    with st.container():
        c3,c4 = st.columns(2)
        with c3:
            c3.header("Info")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        with c4:
            c4.header("Number of null values in each column")
            st.text(df.isnull().sum())

    st.markdown("***")


# Check if you've already initialized the data
if 'df' not in st.session_state:
    # Get the data if you haven't
    df = get_data()
    # Save the data to session state
    st.session_state.df = df
else:
    df = st.session_state.df

make_layout(df)