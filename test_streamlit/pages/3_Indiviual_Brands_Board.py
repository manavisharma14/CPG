import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit_shadcn_ui as ui
from scripts.data import get_data


st.set_page_config(
    page_title="Brand Dashboard",
    page_icon="📊",
    layout="wide",
)

st.header("Brand Dashboard", divider="rainbow")
st.markdown("##")


def plot_trends(df, title):
    fig = px.line(
        df,
        x="Year",
        y="Approx. Total Revenue(INR)",
        template="simple_white",
        title=title,
    )
    # Adjust the size of tick labels and remove axis titles
    fig.update_xaxes(tickfont=dict(size=12), title_text="")
    fig.update_yaxes(tickfont=dict(size=12), title_text="")
    st.plotly_chart(fig, use_container_width=True)


def create_yield_revenue(df, title):
    fig = px.bar(
        df,
        x=df.columns[1],
        y=df.columns[2],
        template="simple_white",
        title=title,
        color=df.columns[1],
    )
    fig.update_layout(
        barmode="stack",
        xaxis={"categoryorder": "total descending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Adjust the size of tick labels and remove axis titles
    fig.update_xaxes(tickfont=dict(size=12), title_text="")
    fig.update_yaxes(tickfont=dict(size=12), title_text="")
    st.plotly_chart(fig, use_container_width=True)


def distribution_chart(df, brand):
    # retail,wholesale,online
    with st.container():
        t1, t2, t3 = st.tabs(
            ["Sales Channel Analysis", "Farm Distribution", "Revenue Trend"]
        )
        with t1:
            c1, c2 = st.columns([3.5, 1])
            with c1:
                fig = px.pie(
                    df,
                    names="Sales Channel",
                    height=600,
                    width=500,
                    hole=0.7,
                    title=f"<b>Sales Channel Distribution for {brand}<b>",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(fig, use_container_width=True)

            with c2:
                # TODO top revenue generating channel
                brand_channels = (
                    df.groupby(["Brand", "Sales Channel"])["Approx. Total Revenue(INR)"]
                    .sum()
                    .reset_index()
                )
                channel_rev_df = (
                    brand_channels.groupby(["Sales Channel"])[
                        "Approx. Total Revenue(INR)"
                    ]
                    .sum()
                    .reset_index()
                )
                top_sale = channel_rev_df["Approx. Total Revenue(INR)"].max()

                ui.metric_card(
                    title=f"Top sale for {brand}",
                    content=f"INR {round(top_sale)}",
                    description="",
                    key="card1",
                )

        with t2:
            selection = ui.switch(
                default_checked=False, label="Revenue", key="switch_visualization"
            )
            chart_df = (
                (
                    df.groupby(["Brand", "Location"])["Quantity (liters/kg)"]
                    .sum()
                    .reset_index()
                    .query("Brand==@brand")
                )
                if not selection
                else (
                    df.groupby(["Brand", "Location"])["Approx. Total Revenue(INR)"]
                    .sum()
                    .reset_index()
                    .query("Brand==@brand")
                )
            )
            title = (
                f"<b>Farm Distribution by Yield</b><br><sup>(liters/kg)</sup>"
                if not selection
                else f"<b>Farm Distribution by Revenue</b><br><sup>INR</sup>"
            )
            create_yield_revenue(chart_df, title)
        # TODO implement Trends linechart
        with t3:
            trends_df = (
                df.groupby(["Brand", "Year"])["Approx. Total Revenue(INR)"]
                .sum()
                .reset_index()
            )
            plot_trends(
                trends_df,
                f"<b>Revenue by Year for {brand}</b><br><sup>in INR</sup>",
            )


def make_charts(df, title):
    fig = px.bar(
        df,
        x=df.columns[0],
        y=df.columns[1],
        template="simple_white",
        title=title,
        text=round(df["Approx. Total Revenue(INR)"]),
        color=df.columns[0],
    )
    fig.update_layout(
        barmode="stack",
        xaxis={"categoryorder": "total descending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Adjust the size of tick labels and remove axis titles
    fig.update_xaxes(tickfont=dict(size=12), title_text="")
    fig.update_yaxes(tickfont=dict(size=12), title_text="")
    st.plotly_chart(fig, use_container_width=True)


def make_layout(df):
    with st.container():

        analysis_type = ui.tabs(
            options=["Month", "Product Category"],
            default_value="Month",
            key="analysis_type",
        )

        # Dropdown for selecting a city
        c1, c2, c3 = st.columns(3)
        with c1:
            brand = ui.select(
                "Select the Brand:",
                options=df["Brand"].unique(),
                key="brand_select",
            )
        with c2:
            state = ui.select(
                "Select the State:",
                options=df["Customer_Location"].unique(),
                key="state_select",
            )
        with c3:
            year = ui.select(
                "Select a Year",
                options=[2019, 2020, 2021, 2022],
                key="year_select",
            )
        fltr_df = df.query(
            "Brand==@brand and Customer_Location==@state and Year==@year"
        )

        if analysis_type == "Product Category":
            fltr = (
                fltr_df.groupby(["Product_Name"], dropna=False)[
                    "Approx. Total Revenue(INR)"
                ]
                .sum()
                .reset_index()
            )
            make_charts(
                fltr,
                f"<b>Product Category sales for {year}</b><br><sup>Sales in INR</sup>",
            )
        else:
            fltr = (
                fltr_df.groupby(["Month"], dropna=False)["Approx. Total Revenue(INR)"]
                .sum()
                .reset_index()
            )
            make_charts(
                fltr, f"<b>Monthly sales for {year}</b><br><sup>Sales in INR</sup>"
            )

        if fltr.empty:
            st.warning(
                "You have not selected any parameter,kindly select filter params."
            )
            st.stop()

        distribution_chart(df.query("Brand==@brand"), brand)


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
