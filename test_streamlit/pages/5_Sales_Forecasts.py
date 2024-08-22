import streamlit as st

st.set_page_config(
    page_title="Brand Sales Forecasts",
    page_icon="📊",
    layout="wide",
)

st.header("Sales Forecasts", divider="rainbow")
st.markdown("##")


def show_forecasts(option):
    st.image(
        f"static/forecasts_uncropped/{option}.jpeg",
        caption=f"Sales forecast for {option} (next 30 days)",
    )


option = st.selectbox(
    "Select the brand for which you want to see forecasts",
    (
        "Amul",
        "Britannia",
        "Dodla",
        "Dynamix",
        "MotherDairy",
        "ParagMilk",
        "Palle2Patnam",
        "Passion Cheese",
        "Raj",
        "Sudha",
        "Warana",
    ),
)

show_forecasts(option)
