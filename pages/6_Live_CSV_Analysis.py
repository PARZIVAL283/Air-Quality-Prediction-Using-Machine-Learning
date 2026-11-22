
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Live CSV Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live Air Quality CSV Analysis")

st.write(
    "Upload an air-quality CSV downloaded from Open-Meteo "
    "to explore pollutant measurements and trends."
)

st.info(
    "This page analyzes API data independently. "
    "It does not generate predictions using the UCI-trained model."
)

uploaded_file = st.file_uploader(
    "Upload Open-Meteo CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        st.subheader("1. Dataset Overview")

        col1, col2 = st.columns(2)

        col1.metric("Total Records", len(df))
        col2.metric("Total Columns", len(df.columns))

        st.write("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Detect available pollutant columns
        pollutant_columns = [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi",
            "european_aqi"
        ]

        available_pollutants = [
            col for col in pollutant_columns
            if col in df.columns
        ]

        if not available_pollutants:
            st.warning(
                "No recognized pollutant columns were found. "
                "Please upload a CSV containing Open-Meteo "
                "air-quality measurements."
            )
            st.stop()

        st.subheader("2. Pollutant Statistics")

        stats = df[available_pollutants].describe().T
        st.dataframe(stats, use_container_width=True)

        st.subheader("3. Pollutant Visualization")

        selected_pollutant = st.selectbox(
            "Select pollutant",
            available_pollutants
        )

        chart_df = df.copy()

        if "time" in chart_df.columns:
            chart_df["time"] = pd.to_datetime(
                chart_df["time"],
                errors="coerce"
            )

            chart_df = chart_df.dropna(subset=["time"])

            fig = px.line(
                chart_df,
                x="time",
                y=selected_pollutant,
                title=f"{selected_pollutant.upper()} Trend",
                markers=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            fig = px.histogram(
                df,
                x=selected_pollutant,
                title=f"{selected_pollutant.upper()} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("4. Download Analyzed Dataset")

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="analyzed_air_quality.csv",
            mime="text/csv"
        )

        st.success("CSV analysis completed successfully!")

    except Exception as e:
        st.error(f"Unable to analyze CSV: {e}")
