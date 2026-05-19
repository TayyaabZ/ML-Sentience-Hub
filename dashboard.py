import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
import plotly.graph_objects as go


st.set_page_config(
    layout="wide",
    page_title="ML SENTIENCE HUB",
    page_icon="⚡",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0a0a0a; color: #00ff41; font-family: monospace; }
    [data-testid="stSidebar"] { background-color: #111111; }
    [data-testid="stSidebar"] * { color: #00ff41; font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='color:#00ff41; font-family:monospace; text-align:center;'>"
    "⚡ ML SENTIENCE HUB</h1>",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder
        .appName("Sentience-Dashboard")
        .master("local[*]")
        .getOrCreate()
    )


@st.cache_resource
def load_aegis_model():
    return PipelineModel.load("models/aegis_model")


@st.cache_resource
def load_nexus_model():
    return PipelineModel.load("models/nexus_model")


spark = get_spark()
aegis_model = load_aegis_model()
nexus_model = load_nexus_model()

st.sidebar.title("⚡ ML SENTIENCE HUB")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigation",
    ["🛡️ AEGIS — Traffic Monitor", "⚡ NEXUS — Grid Predictor"],
)


if section == "🛡️ AEGIS — Traffic Monitor":
    tabs = st.tabs(["Live Detection", "Analytics"])

    with tabs[0]:
        session_duration_sec = st.number_input(
            "Session Duration (seconds)",
            value=120.0,
            step=1.0,
            help="How long the user session lasted. Bots typically complete sessions in under 10 seconds.",
        )
        click_velocity_bps = st.number_input(
            "Click Velocity (requests/sec)",
            value=3.0,
            step=0.1,
            help="Number of page requests per second. Human average is ~2.5, bots average ~18.",
        )
        pages_viewed = st.number_input(
            "Pages Viewed",
            value=4,
            step=1,
            help="Total pages accessed in session. Bots typically scrape 30+ pages.",
        )
        user_agent = st.selectbox(
            "Browser / Client",
            [
                "Mozilla/5.0",
                "Chrome/98",
                "Safari/15",
                "Edge/99",
                "Firefox/97",
                "Opera/80",
                "curl/7.68",
                "python-requests/2.27",
            ],
            help="The HTTP client identifier. curl and python-requests are common bot signatures.",
        )

        if st.button("ANALYZE TRAFFIC"):
            input_df = spark.createDataFrame([
                {
                    "session_duration_sec": float(session_duration_sec),
                    "click_velocity_bps": float(click_velocity_bps),
                    "pages_viewed": int(pages_viewed),
                    "user_agent": user_agent,
                    "class_label": "unknown",
                }
            ])
            prediction = aegis_model.transform(input_df).select("prediction").first()
            if prediction is not None and float(prediction[0]) == 1.0:
                st.error("⚠ BOT DETECTED")
            else:
                st.success("✓ HUMAN VERIFIED")

    with tabs[1]:
        raw_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv("data/aegis_raw_logs.csv")
        )
        pred_df = aegis_model.transform(raw_df)

        total_requests = pred_df.count()
        bot_count = pred_df.filter(col("prediction") == 1.0).count()
        human_count = pred_df.filter(col("prediction") == 0.0).count()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Requests", str(total_requests))
        m2.metric("Bot Count", str(bot_count))
        m3.metric("Human Count", str(human_count))

        importances = [float(x) for x in aegis_model.stages[-1].featureImportances]
        feature_names = [
            "session_duration",
            "click_velocity",
            "pages_viewed",
            "user_agent",
        ]

        fi_fig = go.Figure(
            data=[
                go.Bar(
                    x=importances,
                    y=feature_names,
                    orientation="h",
                    marker_color="#00ff41",
                )
            ]
        )
        fi_fig.update_layout(
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(color="#00ff41"),
            xaxis=dict(title="Importance"),
            yaxis=dict(title="Feature"),
        )
        st.plotly_chart(fi_fig, width="stretch")

        bot_percent = (bot_count / total_requests * 100.0) if total_requests else 0.0
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=bot_percent,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00ff41"},
                    "steps": [
                        {"range": [0, 10], "color": "green"},
                        {"range": [10, 20], "color": "yellow"},
                        {"range": [20, 100], "color": "red"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(color="#00ff41"),
        )
        st.plotly_chart(gauge_fig, width="stretch")

        bot_rows = (
            pred_df.filter(col("prediction") == 1.0)
            .select("timestamp", "ip_address", "click_velocity_bps", "pages_viewed")
            .orderBy(col("timestamp").desc())
            .limit(50)
            .collect()
        )
        st.dataframe([row.asDict() for row in bot_rows], width="stretch")


if section == "⚡ NEXUS — Grid Predictor":
    tabs = st.tabs(["Live Prediction", "Grid Analytics"])

    with tabs[0]:
        kw_draw = st.number_input(
            "Power Draw (kW)",
            value=45.0,
            step=1.0,
            help="Current power consumption. Overload threshold is typically above 90 kW.",
        )
        temperature_c = st.number_input(
            "Grid Temperature (°C)",
            value=38.0,
            step=0.5,
            help="Transformer temperature. Readings above 70°C indicate thermal stress.",
        )
        voltage_drop = st.number_input(
            "Voltage Drop (V)",
            value=5.0,
            step=0.5,
            help="Voltage deviation from nominal. Drops above 25V signal potential overload.",
        )
        sector_id = st.selectbox(
            "City Sector",
            ["F-6", "F-7", "Bahria-Phase-7", "Bahria-Phase-8", "DHA-1", "G-11"],
            help="Islamabad/Rawalpindi grid sector being monitored.",
        )

        if st.button("PREDICT GRID STATUS"):
            input_df = spark.createDataFrame([
                {
                    "kw_draw": float(kw_draw),
                    "temperature_c": float(temperature_c),
                    "voltage_drop": float(voltage_drop),
                    "sector_id": sector_id,
                    "grid_status": "unknown",
                }
            ])
            prediction = nexus_model.transform(input_df).select("prediction").first()
            if prediction is not None and float(prediction[0]) == 1.0:
                st.error("🔴 OVERLOAD PREDICTED")
            else:
                st.success("🟢 GRID STABLE")

    with tabs[1]:
        raw_df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv("data/nexus_raw_telemetry.csv")
        )
        cleaned_df = raw_df.filter(col("timestamp") != "CORRUPTED_TIME")
        pred_df = nexus_model.transform(cleaned_df)

        total_readings = pred_df.count()
        overload_count = pred_df.filter(col("prediction") == 1.0).count()
        normal_count = pred_df.filter(col("prediction") == 0.0).count()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Readings", str(total_readings))
        m2.metric("Overload Count", str(overload_count))
        m3.metric("Normal Count", str(normal_count))

        sector_avg = (
            cleaned_df.groupBy("sector_id")
            .avg("kw_draw")
            .withColumnRenamed("avg(kw_draw)", "avg_kw_draw")
            .collect()
        )
        sectors = [row["sector_id"] for row in sector_avg]
        avg_kw = [float(row["avg_kw_draw"]) for row in sector_avg]

        st.markdown("#### Average Power Draw by Sector")
        st.caption(
            "Uniform distribution reflects synthetic training data. In production this would show real variance across sectors."
        )

        sector_fig = go.Figure(
            data=[
                go.Bar(
                    x=sectors,
                    y=avg_kw,
                    marker_color="#ffaa00",
                )
            ]
        )
        sector_fig.update_layout(
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(color="#00ff41"),
            xaxis=dict(title="sector_id"),
            yaxis=dict(title="avg_kw_draw"),
        )
        st.plotly_chart(sector_fig, width="stretch")

        overload_rows = (
            pred_df.filter(col("prediction") == 1.0)
            .select("timestamp", "sector_id", "kw_draw", "temperature_c", "voltage_drop")
            .collect()
        )
        st.dataframe([row.asDict() for row in overload_rows], width="stretch")
