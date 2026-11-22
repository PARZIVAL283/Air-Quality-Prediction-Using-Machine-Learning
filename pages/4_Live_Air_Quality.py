
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Live Air Quality",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Live Air Quality Monitoring")
st.write(
    "Explore current air-quality conditions and hourly forecasts "
    "for your selected location."
)

st.info(
    "Data is provided by Open-Meteo. These API readings are "
    "displayed separately from predictions made by your trained UCI model."
)

# -----------------------------
# Location selection
# -----------------------------

st.subheader("1. Select Location")

col1, col2 = st.columns(2)

with col1:
    city = st.text_input("Location name", "Dhaka")

with col2:
    timezone = st.selectbox(
        "Timezone",
        ["Asia/Dhaka", "Asia/Kolkata", "UTC"],
        index=0
    )

# Coordinates for selected city
# Add more cities here if needed.
locations = {
    "dhaka": (23.8103, 90.4125),
    "chittagong": (22.3569, 91.7832),
    "chattogram": (22.3569, 91.7832),
    "sylhet": (24.8949, 91.8687),
    "rajshahi": (24.3745, 88.6042),
    "khulna": (22.8456, 89.5403),
    "cumilla": (23.4607, 91.1809),
    "comilla": (23.4607, 91.1809)
}

if city.strip().lower() not in locations:
    st.warning(
        "This city is not in the built-in location list. "
        "Please enter Dhaka, Chittagong, Sylhet, Rajshahi, "
        "Khulna, or Cumilla."
    )
    st.stop()

latitude, longitude = locations[city.strip().lower()]

# -----------------------------
# Fetch API data
# -----------------------------

if st.button("🔄 Get Air Quality Data", type="primary"):

    api_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "us_aqi,european_aqi,pm2_5,pm10,"
            "carbon_monoxide,nitrogen_dioxide,"
            "sulphur_dioxide,ozone"
        ),
        "hourly": (
            "us_aqi,pm2_5,pm10,carbon_monoxide,"
            "nitrogen_dioxide,sulphur_dioxide,ozone"
        ),
        "forecast_days": 3,
        "timezone": timezone
    }

    try:
        with st.spinner("Fetching air-quality data..."):

            response = requests.get(
                api_url,
                params=params,
                timeout=30
            )

            response.raise_for_status()
            data = response.json()

        current = data.get("current", {})
        hourly = data.get("hourly", {})

        st.success(f"Data retrieved for {city.title()}!")

        # -----------------------------
        # Current conditions
        # -----------------------------

        st.subheader("2. Current Air Quality")

        st.caption(
            f"API observation time: {current.get('time', 'Unavailable')}"
        )

        aqi = current.get("us_aqi")

        if aqi is not None:
            st.metric("US Air Quality Index", aqi)

            if aqi <= 50:
                st.success("Good air quality")
            elif aqi <= 100:
                st.info("Moderate air quality")
            elif aqi <= 150:
                st.warning("Unhealthy for sensitive groups")
            elif aqi <= 200:
                st.warning("Unhealthy")
            elif aqi <= 300:
                st.error("Very unhealthy")
            else:
                st.error("Hazardous")

        else:
            st.warning("Current AQI is unavailable.")

        # -----------------------------
        # Pollutant metrics
        # -----------------------------

        st.subheader("3. Pollutant Measurements")

        pollutants = [
            ("PM2.5", "pm2_5", "μg/m³"),
            ("PM10", "pm10", "μg/m³"),
            ("Carbon Monoxide", "carbon_monoxide", "μg/m³"),
            ("Nitrogen Dioxide", "nitrogen_dioxide", "μg/m³"),
            ("Sulphur Dioxide", "sulphur_dioxide", "μg/m³"),
            ("Ozone", "ozone", "μg/m³")
        ]

        cols = st.columns(3)

        for index, (label, key, unit) in enumerate(pollutants):
            value = current.get(key)

            with cols[index % 3]:
                if value is not None:
                    st.metric(label, f"{value:.2f} {unit}")
                else:
                    st.metric(label, "Unavailable")

        # -----------------------------
        # Hourly forecast
        # -----------------------------

        st.subheader("4. Hourly Air Quality Forecast")

        hourly_df = pd.DataFrame(hourly)

        if not hourly_df.empty and "time" in hourly_df.columns:

            hourly_df["time"] = pd.to_datetime(hourly_df["time"])

            selected_pollutant = st.selectbox(
                "Select pollutant to visualize",
                ["pm2_5", "pm10", "carbon_monoxide",
                 "nitrogen_dioxide", "sulphur_dioxide",
                 "ozone", "us_aqi"]
            )

            if selected_pollutant in hourly_df.columns:

                chart = px.line(
                    hourly_df,
                    x="time",
                    y=selected_pollutant,
                    title=f"{selected_pollutant.upper()} — Hourly Forecast",
                    markers=True
                )

                st.plotly_chart(chart, use_container_width=True)

            st.subheader("5. Forecast Data")
            st.dataframe(hourly_df, use_container_width=True)

            st.download_button(
                "⬇️ Download Air Quality Data",
                data=hourly_df.to_csv(index=False).encode("utf-8"),
                file_name="live_air_quality_data.csv",
                mime="text/csv"
            )

        else:
            st.warning("Hourly forecast data is unavailable.")

        st.caption(
            "Source: Open-Meteo Air Quality API. "
            "Forecast data is model-based and is not a direct "
            "ground-station measurement."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"API connection error: {error}")

    except Exception as error:
        st.error(f"Something went wrong: {error}")
