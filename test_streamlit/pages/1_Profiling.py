import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

from scripts.data import get_data

st.set_page_config(
    page_title="Profiling",
    page_icon="📊",
    layout="wide",
)

st.header("Data Profiling")
st.subheader("This is for the exploration of the dataset")


@st.cache_data
def generate_report(df):
    pr = df.profile_report(correlations={"auto": {"calculate": False}})
    return pr

def download_report(report):
    st.download_button(label="Download Full Report", data=report, file_name="report.html")

    

# Check if you've already initialized the data
if "df" not in st.session_state:
    # Get the data if you haven't
    df = get_data()
    # Save the data to session state
    st.session_state.df = df
else:
    df = st.session_state.df


report = generate_report(df)
st_profile_report(report)
download_report(report.to_html())