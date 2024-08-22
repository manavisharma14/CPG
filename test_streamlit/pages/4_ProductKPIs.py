import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit_shadcn_ui as ui
from local_components import card_container
from scripts.data import get_data

st.set_page_config(
    page_title="Product KPIs",
    page_icon="📊",
    layout="wide",
)

st.header("Product KPIs",divider="rainbow")
st.markdown("##")


def sales_KPI(df):
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Product Overview</h1>",
        unsafe_allow_html=True,
    )
    # ** KPIs
    # ** sale,qty,profit,top sold product most popular product
    total_sales_by_product = round(df["Approx. Total Revenue(INR)"].sum())
    total_qty_sold = df["Quantity Sold (liters/kg)"].sum()
    shelf_life = round(df["Shelf Life (days)"].mean())
    min_stock = round(df["Minimum Stock Threshold (liters/kg)"].mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.metric_card(
            title="Total Sales",
            content=f"INR {total_sales_by_product}",
            description="",
            key="card1",
        )
    with c2:
        ui.metric_card(
            title="Total Quantity Sold",
            content=f"{total_qty_sold}",
            description="(liters/kg)",
            key="card2",
        )

    with c3:
        ui.metric_card(
            title="Avg Shelf Life",
            content=f"{shelf_life} days",
            description="",
            key="card3",
        )

    with c4:
        ui.metric_card(
            title="Minimum Stock",
            content=f"{min_stock} ",
            description="(liters/kg)",
            key="card4",
        )

    st.markdown("###")



def make_layout(df):
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            state = ui.select(
                "Select the State:",
                options=df["Customer_Location"].unique(),
                key="state_select",
            )
        with c2:
            brand = ui.select(
                "Select the Brand:",
                options=df["Brand"].unique(),
                key="brand_select",
            )
        with c3:
            product = ui.select(
                "Select the Product:",
                options=df["Product_Name"].unique(),
                key="product",
            )
        st.markdown("###")
        filtered_df = df.query(
            "Customer_Location == @state and Brand == @brand and Product_Name == @product"
        )
        if filtered_df.empty:
            st.warning(
                "You have not selected any parameter,kindly select filter params."
            )
            st.stop()
        sales_KPI(filtered_df)


# Check if you've already initialized the data
if "df" not in st.session_state:
    # Get the data if you haven't
    df = get_data()
    # Save the data to session state
    st.session_state.df = df
else:
    df = st.session_state.df

make_layout(df)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# ! PURGATORY
# st.markdown(
#         "<h1 style='text-align: center; color: white;'>Dataset Overview</h1>",
#         unsafe_allow_html=True,
#     )
#     f3 = df.groupby(["Product_Name"])["Approx. Total Revenue(INR)"].sum().reset_index()
#     f4 = df.groupby(["Customer_Location"])["Approx. Total Revenue(INR)"].sum().reset_index()
#     t1, t2 = st.tabs(["Total Revenue by Product", "Total Revenue by Customer_Location"])
#     with t1:
#         fig = px.bar(
#             f3,
#             x="Product_Name",
#             y="Approx. Total Revenue(INR)",
#             title="Total Revenue by Product",
#             template="simple_white",
#             text=df["Approx. Total Revenue(INR)"].apply(lambda x: f"{x/1000:.1f}"),
#         )
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)",
#         )
#         fig.update_xaxes(tickfont=dict(size=12), title_text="")
#         fig.update_yaxes(tickfont=dict(size=12), title_text="")
#         st.plotly_chart(fig,use_container_width=True)
#     with t2:
#         fig = px.bar(
#             f4,
#             x="Customer_Location",
#             y="Approx. Total Revenue(INR)",
#             title="Total Revenue by Customer_Location",
#             template="simple_white",
#             text=df["Approx. Total Revenue(INR)"].apply(lambda x: f"{x/1000:.1f}"),
#         )
#         fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)",
#         )
#         fig.update_xaxes(tickfont=dict(size=12), title_text="")
#         fig.update_yaxes(tickfont=dict(size=12), title_text="")
#         st.plotly_chart(fig, use_container_width=True)
