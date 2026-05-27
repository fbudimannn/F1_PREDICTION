import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import time
import base64
from src.utils import GRID_2026, CIRCUITS, get_initial_driver_priors, update_elo_ratings, format_lap_time, get_driver_color
from src.data_ingestion import fetch_gp_practice_data, fetch_live_session_timing, fetch_actual_qualifying_results, get_race_status
from src.models import QualifyingModel, MonteCarloSimulator

DRIVER_IMAGE_MAP = {
    "ALB": "albon.avif",
    "ALO": "alonso.avif",
    "BEA": "bearman.avif",
    "BOR": "bortoleto.avif",
    "BOT": "bottas.avif",
    "LEC": "charles.avif",
    "COL": "colapinto.avif",
    "GAS": "gasly.avif",
    "RUS": "george.avif",
    "HUL": "hulkenberg.avif",
    "HAD": "isac.avif",
    "ANT": "kimi.avif",
    "NOR": "landooo.avif",
    "HAM": "lewis.avif",
    "LAW": "liam lawson.avif",
    "LIN": "linbald.avif",
    "VER": "max verstappen.avif",
    "OCO": "ocon.avif",
    "PIA": "oscar.avif",
    "PER": "perez.avif",
    "SAI": "sainz.avif",
    "STR": "stroll.avif"
}

def get_driver_image_base64(driver_code):
    image_file = DRIVER_IMAGE_MAP.get(driver_code)
    if not image_file:
        return ""
    img_path = os.path.join("picture driver", image_file)
    if os.path.exists(img_path):
        try:
            with open(img_path, "rb") as f:
                data = f.read()
                return f"data:image/avif;base64,{base64.b64encode(data).decode()}"
        except Exception:
            return ""
    return ""

def hex_to_rgba(hex_str, alpha):
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def parse_strategy(strat_str):
    mapping = {
        's': 'Soft', 'm': 'Medium', 'h': 'Hard', 'i': 'Intermediate', 'w': 'Wet',
        'soft': 'Soft', 'medium': 'Medium', 'hard': 'Hard', 'intermediate': 'Intermediate', 'wet': 'Wet'
    }
    cleaned = strat_str.replace(',', '-').replace('>', '-').replace(' ', '-')
    tokens = [t.strip().lower() for t in cleaned.split('-') if t.strip()]
    parsed = []
    for t in tokens:
        if t in mapping:
            parsed.append(mapping[t])
        else:
            for char in t:
                if char in mapping:
                    parsed.append(mapping[char])
    if not parsed:
        return ["Medium", "Hard"]
    return parsed

def render_html(html_str):
    cleaned_html = "\n".join(line.lstrip() for line in html_str.splitlines())
    st.markdown(cleaned_html, unsafe_allow_html=True)

def render_sidebar_html(html_str):
    cleaned_html = "\n".join(line.lstrip() for line in html_str.splitlines())
    st.sidebar.markdown(cleaned_html, unsafe_allow_html=True)

# 1. Page Configuration & Layout
st.set_page_config(
    page_title="F1 Bayesian Predictor & Live Tracker",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium CSS Injector (Asphalt Carbon & Neon Red Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stApp {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Target only text and input elements specifically, avoiding generic spans/divs to protect icon webfonts */
    h1, h2, h3, h4, h5, h6, p, label, small, li, a, select, input, button, .stMetric, div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Restore font for Material Symbols & Icons to keep Streamlit collapse/expand arrows rendering correctly */
    .notranslate, [class*="Icon"], [class*="icon"], [class*="stIcon"], [data-testid="stIcon"], [class*="material-icons"] {
        font-family: 'Material Symbols Outlined', 'Material Symbols Rounded', 'Material Symbols Sharp', 'Material Icons' !important;
    }
    
    .stApp {
        background-color: #0d0f12;
        color: #f0f3f6;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #12151c !important;
        border-right: 1px solid #1f2533;
    }
    
    /* Glassmorphism Cards */
    .card, div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(22, 26, 34, 0.7) !important;
        border: 1px solid rgba(255, 24, 1, 0.15) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s ease !important;
    }
    .card:hover, div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(255, 24, 1, 0.4) !important;
        box-shadow: 0 8px 32px 0 rgba(255, 24, 1, 0.1) !important;
    }
    
    /* Neon Red Glowing Accents */
    .neon-text-red {
        color: #ff1801;
        text-shadow: 0 0 10px rgba(255, 24, 1, 0.5);
        font-weight: 800;
    }
    .neon-text-cyan {
        color: #00e5ff;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
        font-weight: 800;
    }
    
    /* Custom headers */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Metric Card Custom Overrides */
    div[data-testid="stMetricValue"] {
        color: #f0f3f6;
        font-weight: 800;
        font-size: 2rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #8f9cae;
        font-weight: 400;
    }
    
    /* Custom Strategy badges */
    .strategy-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 5px;
    }
    .badge-soft { background-color: #ff1801; color: white; }
    .badge-medium { background-color: #ffcc00; color: black; }
    .badge-hard { background-color: #ffffff; color: black; }
    .badge-intermediate { background-color: #00e5ff; color: black; }
    .badge-wet { background-color: #0090ff; color: white; }
    
    /* Race Status Badges */
    .badge-done {
        display: inline-block;
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0, 200, 83, 0.3);
    }
    .badge-ongoing {
        display: inline-block;
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(255, 23, 68, 0.4);
        animation: pulse-badge 1.5s ease-in-out infinite;
    }
    .badge-soon {
        display: inline-block;
        background: linear-gradient(135deg, #ffab00, #ffd740);
        color: #1a1a1a;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(255, 171, 0, 0.3);
    }
    @keyframes pulse-badge {
        0%, 100% { opacity: 1; box-shadow: 0 2px 8px rgba(255, 23, 68, 0.4); }
        50% { opacity: 0.7; box-shadow: 0 2px 16px rgba(255, 23, 68, 0.7); }
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
if os.path.exists("logo f1.png"):
    st.sidebar.image("logo f1.png", use_container_width=True)
st.sidebar.markdown(f"<h1 class='neon-text-red' style='font-size: 28px; margin-top: 10px; margin-bottom: 20px;'>F1 PREDICTOR Pro</h1>", unsafe_allow_html=True)

# Circuit Selection
active_circuit = st.sidebar.selectbox(
    "Select Active Circuit:",
    options=list(CIRCUITS.keys()),
    format_func=lambda x: CIRCUITS[x]["name"]
)
circuit_data = CIRCUITS[active_circuit]

# Fetch race status for active circuit (DONE / ONGOING / SOON)
@st.cache_data(ttl=30)
def get_cached_race_status(circuit_id):
    return get_race_status(circuit_id)

race_status = get_cached_race_status(active_circuit)

# Display race status badge in sidebar
st.sidebar.markdown("---")
if race_status["status"] == "DONE":
    st.sidebar.markdown(f"<span class='badge-done'>✅ RACE COMPLETED</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<small style='color: #8f9cae;'>Finished on {race_status['event_date']} — Full data available ({race_status['latest_lap']}/{race_status['total_laps']} laps)</small>", unsafe_allow_html=True)
elif race_status["status"] == "ONGOING":
    st.sidebar.markdown(f"<span class='badge-ongoing'>🔴 LIVE — LAP {race_status['latest_lap']}/{race_status['total_laps']}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<small style='color: #ff5252;'>Race in progress — Auto-refreshing every 30s</small>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"<span class='badge-soon'>📅 UPCOMING</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<small style='color: #ffd740;'>Race Day: {race_status['event_date']}</small>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📍 Circuit Specifications")
st.sidebar.markdown(f"**Laps:** {circuit_data['laps']} | **Length:** {circuit_data['length_km']} km")
st.sidebar.markdown(f"**Safety Car Rate:** {int(circuit_data['sc_probability']*100)}%")
st.sidebar.markdown(f"**Overtaking:** {'Easy/Moderate' if circuit_data['overtaking_index'] > 0.5 else 'Very Hard'}")

st.sidebar.markdown("---")
st.sidebar.markdown(f"### ☁️ Environmental Settings")
weather_condition = st.sidebar.selectbox(
    "Weather Condition:",
    options=["Dry", "Damp / Light Rain", "Wet / Heavy Rain"],
    index=0
)

# Map weather to rain_intensity and default track temp
rain_intensity = 0.0
default_temp = 28
if weather_condition == "Damp / Light Rain":
    rain_intensity = 0.5
    default_temp = 20
elif weather_condition == "Wet / Heavy Rain":
    rain_intensity = 1.0
    default_temp = 16

# Keep track of track_temp globally using session state to prevent tab unmounting bugs
if "track_temp" not in st.session_state or st.session_state.get("prev_weather") != weather_condition:
    st.session_state.track_temp = default_temp
    st.session_state.prev_weather = weather_condition

track_temp = st.sidebar.slider(
    "Track Temperature (°C):", 
    min_value=15, 
    max_value=55, 
    value=int(st.session_state.track_temp), 
    step=1,
    key="track_temp_slider"
)
st.session_state.track_temp = track_temp

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Operations")
if st.sidebar.button("Refresh Data & Clear Cache", help="Clears Streamlit cache and forces a full model retrain and data fetch"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<small style='color: #8f9cae;'>F1 Bayesian Predictor v1.0.0<br>2026 Grid Simulation</small>", unsafe_allow_html=True)

# 4. Main Panel Page Title
st.markdown(f"<h1 style='margin-bottom: 5px;'>F1 Bayesian Predictor & Live Tracker <span class='neon-text-red'>2026</span></h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #8f9cae; font-size: 16px; margin-bottom: 25px;'>Actively tracking and simulating: <strong>GP {circuit_data['name']}</strong></p>", unsafe_allow_html=True)

# 5. Core Model & Session State Initialization
@st.cache_resource
def load_qualy_model_v2():
    model = QualifyingModel()
    model.train_mock_models()
    return model

qualy_model = load_qualy_model_v2()

# Cached function for ingestion to prevent constant API/generation overhead on minor reruns
@st.cache_data
def get_cached_practice_data_v2(circuit_id):
    return fetch_gp_practice_data(circuit_id, session="FP3")
# Cached function for actual qualifying data to prevent API overhead on minor reruns
@st.cache_data
def get_cached_actual_qualifying_results(circuit_id):
    return fetch_actual_qualifying_results(circuit_id)

# Ingest practice times for the track
fp3_times, speed_traps, tyre_codes, loaded_session = get_cached_practice_data_v2(active_circuit)

# Ensure data directory exists and persist API practice data locally
os.makedirs("data", exist_ok=True)
practice_export = {
    "circuit": active_circuit,
    "fp3_times": fp3_times,
    "speed_traps": speed_traps,
    "tyre_codes": tyre_codes
}
with open("data/fp3_practice_data.json", "w") as f:
    json.dump(practice_export, f, indent=4)

# Initialize driver priors (Elo)
if "driver_priors" not in st.session_state:
    st.session_state.driver_priors = get_initial_driver_priors()

# Dynamic Bayesian Elo Rating Update based on practice head-to-heads
# Resetting to base priors before updating to prevent cumulative drift on every page rerun
st.session_state.driver_priors = update_elo_ratings(get_initial_driver_priors(), fp3_times)

# Calculate qualifying results globally so they are always available and persistent across tab switches
qualy_results = qualy_model.predict_qualifying(
    st.session_state.driver_priors, fp3_times, track_temp, speed_traps, tyre_codes, rain_intensity=rain_intensity
)

# Persist qualifying predictions to local data folder
os.makedirs("data", exist_ok=True)
with open("data/qualifying_predictions.json", "w") as f:
    json.dump(qualy_results, f, indent=4)

# 6. Tab Layouts
tab_telemetry, tab_qualy, tab_race = st.tabs([
    "📊 LIVE PRACTICE & TELEMETRY", 
    "⏱️ QUALIFYING GRID PREDICTOR", 
    "🏁 MONTE CARLO RACE SIMULATOR"
])

# ==========================================
# TAB 1: LIVE PRACTICE & TELEMETRY COMPARATOR
# ==========================================
with tab_telemetry:
    st.markdown(f"### 📊 {loaded_session} Telemetry Analysis")
    st.markdown(f"Compare dynamic lap telemetry (speed, throttle, brake profiles) set during {loaded_session}.")
    
    col_t1, col_t2 = st.columns([1, 3])
    
    with col_t1:
        with st.container(border=True):
            st.markdown("#### Driver Comparators")
            driver_a = st.selectbox("Driver 1:", options=list(GRID_2026.keys()), index=0)
            driver_b = st.selectbox("Driver 2:", options=list(GRID_2026.keys()), index=2)
            
            # Get driver images and colors
            img_a = get_driver_image_base64(driver_a)
            img_b = get_driver_image_base64(driver_b)
            color_a = get_driver_color(driver_a)
            color_b = get_driver_color(driver_b)
            
            # Show a beautiful VS layout with pictures
            render_html(f"""
<div style="display: flex; justify-content: space-around; align-items: center; margin-top: 15px; margin-bottom: 15px; background: rgba(22, 26, 34, 0.5); padding: 15px 10px; border-radius: 10px; border: 1px solid rgba(255, 24, 1, 0.15); box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
    <div style="text-align: center; width: 45%;">
        <div style="position: relative; display: inline-block;">
            <img src="{img_a}" style="width: 65px; height: 65px; border-radius: 50%; border: 3px solid {color_a}; object-fit: cover; object-position: center top; box-shadow: 0 0 12px {color_a}88; background-color: #12151c;">
        </div>
        <div style="font-weight: 800; font-size: 13px; margin-top: 6px; color: #f0f3f6;">{driver_a}</div>
        <div style="font-size: 10px; color: #8f9cae; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{GRID_2026[driver_a]['name']}</div>
    </div>
    <div style="font-weight: 800; font-size: 16px; color: #ff1801; text-shadow: 0 0 10px rgba(255,24,1,0.6); font-family: 'Poppins', sans-serif;">VS</div>
    <div style="text-align: center; width: 45%;">
        <div style="position: relative; display: inline-block;">
            <img src="{img_b}" style="width: 65px; height: 65px; border-radius: 50%; border: 3px solid {color_b}; object-fit: cover; object-position: center top; box-shadow: 0 0 12px {color_b}88; background-color: #12151c;">
        </div>
        <div style="font-weight: 800; font-size: 13px; margin-top: 6px; color: #f0f3f6;">{driver_b}</div>
        <div style="font-size: 10px; color: #8f9cae; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{GRID_2026[driver_b]['name']}</div>
    </div>
</div>
""")
            
            st.markdown(f"<br><strong>{loaded_session} Average Lap Times:</strong>", unsafe_allow_html=True)
            st.write(f"🏎️ **{driver_a}:** {format_lap_time(fp3_times[driver_a])}")
            st.write(f"🏎️ **{driver_b}:** {format_lap_time(fp3_times[driver_b])}")
            st.write(f"⏱️ **Delta:** {abs(fp3_times[driver_a] - fp3_times[driver_b]):.3f}s")
        
    with col_t2:
        # Generate elegant telemetry charts (Speed Curves)
        np.random.seed(12)
        distance = np.linspace(0, circuit_data["length_km"] * 1000, 300)
        
        # Telemetry curves (chicanes and straights)
        speed_base = 280 + 60 * np.sin(distance / 220) - 40 * np.sin(distance / 80)
        # Apply minor variation for driver A and B based on speed traps
        speed_a = speed_base + (speed_traps[driver_a] - 330) * 0.5 + np.random.normal(0, 2, 300)
        speed_b = speed_base + (speed_traps[driver_b] - 330) * 0.5 + np.random.normal(0, 2, 300)
        
        # Clip values to realistic ranges
        speed_a = np.clip(speed_a, 80, 350)
        speed_b = np.clip(speed_b, 80, 350)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=distance, y=speed_a, name=f"{GRID_2026[driver_a]['name']} ({driver_a})", line=dict(color=get_driver_color(driver_a), width=3)))
        fig.add_trace(go.Scatter(x=distance, y=speed_b, name=f"{GRID_2026[driver_b]['name']} ({driver_b})", line=dict(color=get_driver_color(driver_b), width=3, dash='dash')))
        
        fig.update_layout(
            title="Telemetry Speed Curve Comparison (Circuit Lap)",
            xaxis_title="Distance along track (meters)",
            yaxis_title="Speed (km/h)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: QUALIFYING GRID PREDICTOR
# ==========================================
with tab_qualy:
    st.markdown("### ⏱️ Qualifying Grid Predictor (Learning-To-Rank)")
    st.markdown("Predicting relative starting positions using LightGBM Ranker and Quantile lap-time intervals.")
    
    col_q1, col_q2 = st.columns([1, 2])
    
    with col_q1:
        with st.container(border=True):
            st.markdown("#### Qualifying Weather Parameters")
            st.metric("Track Temperature", f"{track_temp} °C")
            
            st.markdown("<br><strong>Dynamic Form Influence:</strong>", unsafe_allow_html=True)
            st.markdown(f"Priors are updated dynamically using {loaded_session} sector paces. Click below to recalculate:")
            if st.button("Force Recalculate Prior Elo"):
                st.session_state.driver_priors = update_elo_ratings(get_initial_driver_priors(), fp3_times)
                st.success("Driver Bayesian prior ratings updated successfully!")
        
    with col_q2:
        df_qualy = pd.DataFrame(qualy_results)
        
        # Plot predicted qualifying times (Quantile intervals) using full driver names
        fig_q = go.Figure()
        
        colors_list = [get_driver_color(code) for code in df_qualy["driver_code"]]

        # Render confidence intervals as error bars or custom ranges
        fig_q.add_trace(go.Scatter(
            x=df_qualy["driver_name"],
            y=df_qualy["median_time"],
            mode='markers',
            name='Predicted Median Time',
            marker=dict(color=colors_list, size=10),
            error_y=dict(
                type='data',
                symmetric=False,
                array=df_qualy["worst_case_time"] - df_qualy["median_time"],
                arrayminus=df_qualy["median_time"] - df_qualy["best_case_time"],
                color='#8f9cae'
            )
        ))
        
        fig_q.update_layout(
            title="Estimated Q3 Lap Times with 90% Bayesian Credible Intervals",
            xaxis_title="Driver Name",
            yaxis_title="Lap Time (seconds)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_q, use_container_width=True)
        
    # Render final grid table
    st.markdown("#### Predicted Q3 Final Starting Grid")
    
    grid_cols = st.columns(4)
    for idx, p in enumerate(qualy_results): # Display all predicted drivers on the grid
        col_idx = idx % 4
        with grid_cols[col_idx]:
            # Generate simulated SHAP text
            shap_fp3 = -0.08 if p["predicted_position"] <= 3 else 0.04
            shap_elo = -0.05 if st.session_state.driver_priors[p["driver_code"]] > 1700 else 0.02
            d_color = get_driver_color(p["driver_code"])
            driver_img = get_driver_image_base64(p["driver_code"])
            
            render_html(f"""
<div class="card" style="border-left: 5px solid {d_color}; position: relative; overflow: hidden; min-height: 180px;">
    <!-- Faded Driver Portrait Background -->
    <div style="position: absolute; right: -12px; bottom: 0px; width: 130px; height: 160px; background-image: url('{driver_img}'); background-size: cover; background-position: center top; opacity: 0.28; mask-image: linear-gradient(to left, rgba(0,0,0,1) 15%, rgba(0,0,0,0) 80%); -webkit-mask-image: linear-gradient(to left, rgba(0,0,0,1) 15%, rgba(0,0,0,0) 80%); pointer-events: none; z-index: 0;"></div>
    <div style="position: relative; z-index: 1;">
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 28px; font-weight: 800; color: {d_color};">P{p['predicted_position']}</span>
            <span style="font-size: 14px; font-weight: 600; color: #8f9cae;">{p['driver_code']}</span>
        </div>
        <h5 style="margin: 3px 0; font-weight: 800; font-size: 16px; color: #f0f3f6;">{p['driver_name']}</h5>
        <small style="color: #8f9cae;">{p['team']}</small><br>
        <div style="width: 45%; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 10px; margin-bottom: 8px;"></div>
        <div style="font-size: 11px; color: #f0f3f6; line-height: 1.45;">
            <div>🎯 <strong>Median:</strong> {format_lap_time(p['median_time'])}</div>
            <div style="margin-top: 1px;">🔍 <strong>Range:</strong> {format_lap_time(p['best_case_time'])} - {format_lap_time(p['worst_case_time'])}</div>
        </div>
        <div style="margin-top: 4px; font-size: 9px; color: {d_color}; opacity: 0.9;">
            ✨ <strong>SHAP:</strong> FP3 ({shap_fp3:.2f}s) | Prior ({shap_elo:.2f}s)
        </div>
    </div>
</div>
""")

# ==========================================
# TAB 3: MONTE CARLO RACE SIMULATOR
# ==========================================
with tab_race:
    st.markdown("### 🏁 Vectorized Monte Carlo Race Simulator (10,000 runs)")
    st.markdown("Running physical, stateful lap-by-lap simulations to predict final outcomes and risk distributions.")
    
    # Setup live session state if chosen
    starting_grid = [p["driver_code"] for p in qualy_results]
    active_state = None
    curr_lap = 0
    grid_source_used = "ML Prediction"
    
    # Determine dynamic lap target based on race status
    if race_status["status"] == "ONGOING" and race_status["latest_lap"] is not None:
        mid_lap = race_status["latest_lap"]
    elif race_status["status"] == "DONE" and race_status["latest_lap"] is not None:
        mid_lap = circuit_data["laps"] // 2  # Default to midpoint for replay
    else:
        mid_lap = circuit_data["laps"] // 2
    
    # Control Options
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    
    with col_c1:
        with st.container(border=True):
            st.markdown("#### Simulation Controls")
            sim_iterations = st.selectbox("Simulation Runs:", [5000, 10000, 20000], index=1)
            
            # Build mode options based on race status
            if race_status["status"] == "ONGOING":
                # Force live mode with badge
                st.markdown(f"<span class='badge-ongoing'>🔴 LIVE — LAP {mid_lap}/{circuit_data['laps']}</span>", unsafe_allow_html=True)
                sim_mode = st.radio("Simulation Mode:", [
                    f"🔴 Live Race Tracking (Lap {mid_lap})",
                    "Standard Pre-Race Predictor"
                ])
            elif race_status["status"] == "DONE":
                # Allow user to pick any lap via slider
                st.markdown(f"<span class='badge-done'>✅ COMPLETED</span>", unsafe_allow_html=True)
                sim_mode = st.radio("Simulation Mode:", [
                    "Standard Pre-Race Predictor",
                    "Replay Mid-Race State"
                ])
                if sim_mode == "Replay Mid-Race State":
                    mid_lap = st.slider(
                        "Select Lap to Replay:",
                        min_value=1,
                        max_value=race_status["latest_lap"],
                        value=circuit_data["laps"] // 2,
                        step=1
                    )
            else:
                # SOON: only pre-race mode available
                st.markdown(f"<span class='badge-soon'>📅 UPCOMING</span>", unsafe_allow_html=True)
                st.caption(f"Race Day: {race_status['event_date']}")
                sim_mode = "Standard Pre-Race Predictor"
                st.info("🔒 Live tracking will be available once the race starts.")
            
            if sim_mode == "Standard Pre-Race Predictor":
                grid_source = st.radio(
                    "Starting Grid Source:",
                    ["Use Predicted Qualifying Grid (ML)", "Use Actual Saturday Qualifying Classification (API)"],
                    index=0
                )
            else:
                grid_source = "Use Live Running Order"
            
    # Determine if we are in live/replay mid-race mode
    is_live_mode = sim_mode in [f"🔴 Live Race Tracking (Lap {mid_lap})", "Replay Mid-Race State"]
    
    if is_live_mode:
        curr_lap = mid_lap
        active_state = fetch_live_session_timing(active_circuit, active_lap=mid_lap)
        starting_grid = active_state["sorted_drivers"]
        data_src = active_state.get("data_source", "Unknown")
        grid_source_used = f"Live Lap {mid_lap} ({data_src})"
    elif grid_source == "Use Actual Saturday Qualifying Classification (API)":
        actual_grid = get_cached_actual_qualifying_results(active_circuit)
        if actual_grid:
            starting_grid = actual_grid
            grid_source_used = "Actual Sat Qualy (API)"
        else:
            # Fallback to a realistic grid based on updated Bayesian Elo ratings
            np.random.seed(99)
            actual_fallback = sorted(
                list(GRID_2026.keys()), 
                key=lambda x: st.session_state.driver_priors.get(x, 1500) + np.random.normal(0, 25), 
                reverse=True
            )
            starting_grid = actual_fallback
            grid_source_used = "Actual Sat Qualy (Simulated Fallback)"
        
    with col_c2:
        with st.container(border=True):
            st.markdown("#### Strategy Configurator (Top 5 Grid)")
            # Customize tyre strategy for top drivers dynamically
            top_5_strats = {}
            for idx, d in enumerate(starting_grid[:5]):
                driver_name = GRID_2026[d]["name"]
                # Default selection based on driver
                default_idx = 0
                if d == "HAM":
                    default_idx = 1
                elif d == "VER":
                    default_idx = 0
                else:
                    default_idx = idx % 3
                
                strategy_options = [
                    "Medium-Hard", 
                    "Soft-Medium-Medium", 
                    "Medium-Medium-Hard", 
                    "Soft-Hard",
                    "Intermediate-Intermediate",
                    "Wet-Intermediate",
                    "Intermediate-Medium",
                    "Soft-Intermediate-Wet",
                    "🔧 Custom Strategy..."
                ]
                
                selected_opt = st.selectbox(
                    f"{d} ({driver_name}) Strategy:", 
                    strategy_options, 
                    index=default_idx,
                    key=f"strat_select_{d}"
                )
                
                if selected_opt == "🔧 Custom Strategy...":
                    custom_input = st.text_input(
                        f"Enter custom strategy for {d} (e.g., S-M-M, M-H):", 
                        value="Medium-Hard",
                        key=f"strat_custom_{d}"
                    )
                    top_5_strats[d] = custom_input
                else:
                    top_5_strats[d] = selected_opt
                
                # Dynamic visual feedback of parsed strategy
                parsed_compounds = parse_strategy(top_5_strats[d])
                badge_html = ""
                for compound in parsed_compounds:
                    c_class = f"badge-{compound.lower()}"
                    badge_html += f"<span class='strategy-badge {c_class}'>{compound}</span>"
                st.markdown(f"<div style='margin-bottom: 12px;'>{badge_html}</div>", unsafe_allow_html=True)
        
    with col_c3:
        with st.container(border=True):
            st.markdown("#### Circuit Event Injectors")
            sc_toggle = st.slider("Safety Car Probability Multiplier:", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            dnf_toggle = st.slider("DNF Baseline Multiplier:", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        
    # Compile strategies map dynamically
    tyre_strategies = {}
    for d in starting_grid:
        if d in top_5_strats:
            tyre_strategies[d] = parse_strategy(top_5_strats[d])
        else:
            # Default strategy based on grid position
            tyre_strategies[d] = ["Medium", "Hard"]
        
    # Render starting grid source info badge
    st.info(f"🚦 **Starting Grid Status:** Using **{grid_source_used}** as starting order")

    # Execute simulation
    sim_meta = CIRCUITS[active_circuit]
    # Apply sliders to metadata
    sim_meta["sc_probability"] *= sc_toggle
    sim_meta["base_dnf_probability"] *= dnf_toggle
    
    sim_engine = MonteCarloSimulator(active_circuit)
    
    with st.spinner(f"Running {sim_iterations:,} vectorized NumPy race runs..."):
        sim_stats = sim_engine.simulate_race(
            starting_grid, tyre_strategies, num_sims=sim_iterations, current_lap=curr_lap, active_state=active_state, rain_intensity=rain_intensity
        )
        
    # Persist race simulation results to local data folder
    with open("data/race_simulations.json", "w") as f:
        json.dump(sim_stats, f, indent=4)
        
    # Display results
    df_sim = pd.DataFrame.from_dict(sim_stats, orient='index').reset_index().rename(columns={"index": "driver_code"})
    df_sim_sorted = df_sim.sort_values("win_probability", ascending=False)
    
    # 3D Visual Podium for P1, P2, P3 Simulated Winners
    if len(df_sim_sorted) >= 3:
        p1_row = df_sim_sorted.iloc[0]
        p2_row = df_sim_sorted.iloc[1]
        p3_row = df_sim_sorted.iloc[2]
        
        p1_code = p1_row["driver_code"]
        p2_code = p2_row["driver_code"]
        p3_code = p3_row["driver_code"]
        
        p1_name = p1_row["driver_name"]
        p2_name = p2_row["driver_name"]
        p3_name = p3_row["driver_name"]
        
        p1_prob = p1_row["win_probability"]
        p2_prob = p2_row["win_probability"]
        p3_prob = p3_row["win_probability"]
        
        p1_img = get_driver_image_base64(p1_code)
        p2_img = get_driver_image_base64(p2_code)
        p3_img = get_driver_image_base64(p3_code)
        
        p1_color = get_driver_color(p1_code)
        p2_color = get_driver_color(p2_code)
        p3_color = get_driver_color(p3_code)
        
        render_html(f"""
<div style="display: flex; justify-content: center; align-items: flex-end; gap: 15px; margin: 35px auto 25px auto; max-width: 600px; height: 260px; font-family: 'Poppins', sans-serif;">
    <!-- P2 (Left) -->
    <div style="display: flex; flex-direction: column; align-items: center; width: 30%; max-width: 140px;">
        <div style="position: relative; text-align: center; margin-bottom: 5px;">
            <img src="{p2_img}" style="width: 70px; height: 70px; border-radius: 50%; border: 3px solid {p2_color}; object-fit: cover; object-position: center top; box-shadow: 0 4px 15px {p2_color}66; background-color: #12151c;">
            <span style="position: absolute; bottom: -5px; right: 5px; background: #c0c0c0; color: #111; font-weight: 800; font-size: 10px; padding: 2px 6px; border-radius: 10px;">P2</span>
        </div>
        <div style="font-weight: 800; font-size: 13px; color: #f0f3f6; margin-top: 2px; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%;">{p2_name}</div>
        <div style="font-size: 11px; color: #8f9cae; margin-bottom: 8px;">{p2_prob:.1f}% Win</div>
        <div style="width: 100%; height: 75px; background: linear-gradient(180deg, rgba(90,100,112,0.3), rgba(44,50,58,0.3)); border-radius: 8px 8px 0 0; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(192,192,192,0.25); border-bottom: none; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <span style="font-size: 28px; font-weight: 800; color: #c0c0c0; text-shadow: 0 0 10px rgba(192,192,192,0.3);">2</span>
        </div>
    </div>
    <!-- P1 (Center) -->
    <div style="display: flex; flex-direction: column; align-items: center; width: 34%; max-width: 160px; transform: translateY(-15px);">
        <div style="position: relative; text-align: center; margin-bottom: 5px;">
            <div style="position: absolute; top: -20px; left: 50%; transform: translateX(-50%) rotate(-5deg); font-size: 20px; z-index: 2;">👑</div>
            <img src="{p1_img}" style="width: 90px; height: 90px; border-radius: 50%; border: 4px solid {p1_color}; object-fit: cover; object-position: center top; box-shadow: 0 4px 20px {p1_color}bb; background-color: #12151c;">
            <span style="position: absolute; bottom: -5px; right: 8px; background: #ffd700; color: #111; font-weight: 800; font-size: 11px; padding: 2px 7px; border-radius: 10px; box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);">P1</span>
        </div>
        <div style="font-weight: 800; font-size: 15px; color: #f0f3f6; margin-top: 2px; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%;">{p1_name}</div>
        <div style="font-size: 12px; color: #ff1801; font-weight: 800; text-shadow: 0 0 10px rgba(255,24,1,0.4); margin-bottom: 8px;">{p1_prob:.1f}% Win</div>
        <div style="width: 100%; height: 105px; background: linear-gradient(180deg, rgba(255,24,1,0.25), rgba(107,10,0,0.25)); border-radius: 8px 8px 0 0; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,24,1,0.4); border-bottom: none; box-shadow: 0 4px 25px rgba(255,24,1,0.25);">
            <span style="font-size: 38px; font-weight: 800; color: #ffd700; text-shadow: 0 0 15px rgba(255,215,0,0.6);">1</span>
        </div>
    </div>
    <!-- P3 (Right) -->
    <div style="display: flex; flex-direction: column; align-items: center; width: 30%; max-width: 140px;">
        <div style="position: relative; text-align: center; margin-bottom: 5px;">
            <img src="{p3_img}" style="width: 70px; height: 70px; border-radius: 50%; border: 3px solid {p3_color}; object-fit: cover; object-position: center top; box-shadow: 0 4px 15px {p3_color}66; background-color: #12151c;">
            <span style="position: absolute; bottom: -5px; right: 5px; background: #cd7f32; color: #111; font-weight: 800; font-size: 10px; padding: 2px 6px; border-radius: 10px;">P3</span>
        </div>
        <div style="font-weight: 800; font-size: 13px; color: #f0f3f6; margin-top: 2px; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%;">{p3_name}</div>
        <div style="font-size: 11px; color: #8f9cae; margin-bottom: 8px;">{p3_prob:.1f}% Win</div>
        <div style="width: 100%; height: 55px; background: linear-gradient(180deg, rgba(160,90,44,0.3), rgba(74,39,17,0.3)); border-radius: 8px 8px 0 0; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(205,127,50,0.25); border-bottom: none; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <span style="font-size: 24px; font-weight: 800; color: #cd7f32; text-shadow: 0 0 10px rgba(205,127,50,0.3);">3</span>
        </div>
    </div>
</div>
""")
    
    st.markdown("#### Predicted Race Finish Probability (Top 10)")
    
    # Plotly probability bar chart using beautiful full driver names
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_sim_sorted["driver_name"][:10],
        y=df_sim_sorted["win_probability"][:10],
        name="Win Chance %",
        marker_color='#ff1801'
    ))
    fig_bar.add_trace(go.Bar(
        x=df_sim_sorted["driver_name"][:10],
        y=df_sim_sorted["podium_probability"][:10],
        name="Podium Chance %",
        marker_color='#00e5ff'
    ))
    fig_bar.add_trace(go.Bar(
        x=df_sim_sorted["driver_name"][:10],
        y=df_sim_sorted["top10_probability"][:10],
        name="Top 10 Chance %",
        marker_color='#8f9cae'
    ))
    
    fig_bar.update_layout(
        barmode='group',
        title="Simulated Finish Probabilities per Driver",
        xaxis_title="Driver Name",
        yaxis_title="Probability (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Probability breakdown table
    st.markdown("#### Full Probability Breakdown Table")
    
    # Color team and driver code cells with their constructor palette
    def style_team_column(row):
        color = get_driver_color(row["driver_code"])
        hex_str = color.lstrip('#')
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "#0d0f12" if brightness > 150 else "#f0f3f6"
        
        styles = [''] * len(row)
        cols = list(row.index)
        if "driver_code" in cols:
            styles[cols.index("driver_code")] = f"background-color: {color}; color: {text_color}; font-weight: bold;"
        if "team" in cols:
            styles[cols.index("team")] = f"background-color: {color}; color: {text_color}; font-weight: bold;"
        return styles

    st.dataframe(
        df_sim_sorted[["driver_code", "driver_name", "team", "win_probability", "podium_probability", "top10_probability", "dnf_probability"]].style.apply(style_team_column, axis=1),
        column_config={
            "driver_code": st.column_config.TextColumn("Code"),
            "driver_name": st.column_config.TextColumn("Driver Name"),
            "team": st.column_config.TextColumn("Team"),
            "win_probability": st.column_config.ProgressColumn(
                "Win Prob",
                help="Probability of winning the race",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "podium_probability": st.column_config.ProgressColumn(
                "Podium Prob",
                help="Probability of finishing on the podium",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "top10_probability": st.column_config.ProgressColumn(
                "Top 10 Prob",
                help="Probability of finishing in the top 10",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "dnf_probability": st.column_config.ProgressColumn(
                "DNF Prob",
                help="Probability of retiring from the race",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            )
        },
        use_container_width=True,
        hide_index=True
    )

# 7. Live Race Auto-Refresh (ONGOING status only)
if race_status["status"] == "ONGOING":
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
        
    @st.fragment(run_every=30)
    def live_auto_refresh():
        # Check if 30 seconds (with a minor 2s buffer) have elapsed since the last full page load
        if time.time() - st.session_state.last_refresh >= 28:
            st.session_state.last_refresh = time.time()
            st.rerun()
            
    live_auto_refresh()

# 8. F1 Soundtrack — Background Music Player (Sidebar)
# Placed in the sidebar because Streamlit's main content area has
# overflow:hidden + transform that breaks position:fixed CSS.
# The sidebar is already fixed-position, so the player is always visible.

_mp3_path = os.path.join("static", "F1.mp3")
if os.path.exists(_mp3_path):
    with open(_mp3_path, "rb") as _f:
        _audio_bytes = _f.read()

    st.sidebar.markdown("---")
    render_sidebar_html("""
    <div style="
        background: linear-gradient(145deg, rgba(13,15,18,0.95), rgba(30,30,42,0.95));
        border: 1px solid rgba(255,24,1,0.35);
        border-radius: 14px;
        padding: 14px 16px 12px 16px;
        box-shadow: 0 0 25px rgba(255,24,1,0.08), 0 4px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        margin-bottom: 8px;
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,24,1,0.15);
        ">
            <span style="font-size: 16px;">🏎️</span>
            <span style="
                font-family: 'Poppins', sans-serif;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 2.5px;
                color: #ff1801;
                text-shadow: 0 0 12px rgba(255,24,1,0.5);
                text-transform: uppercase;
            ">F1 SOUNDTRACK</span>
            <span style="
                margin-left: auto;
                width: 6px;
                height: 6px;
                background: #ff1801;
                border-radius: 50%;
                box-shadow: 0 0 8px rgba(255,24,1,0.8);
                animation: f1pulse 1.5s ease-in-out infinite;
            "></span>
        </div>
        <style>
            @keyframes f1pulse {
                0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(255,24,1,0.8); }
                50% { opacity: 0.3; box-shadow: 0 0 4px rgba(255,24,1,0.3); }
            }
        </style>
    </div>
    """)

    # Inject CSS to style the sidebar audio element
    st.markdown("""
    <style>
        /* Style the sidebar audio player */
        section[data-testid="stSidebar"] audio {
            width: 100% !important;
            height: 38px !important;
            border-radius: 10px !important;
            outline: none !important;
        }

        /* Webkit Shadow DOM: controls panel */
        section[data-testid="stSidebar"] audio::-webkit-media-controls-panel {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 10px !important;
        }

        /* Webkit Shadow DOM: play button red */
        section[data-testid="stSidebar"] audio::-webkit-media-controls-play-button {
            background-color: #ff1801 !important;
            border-radius: 50% !important;
            transform: scale(1.15) !important;
        }
        section[data-testid="stSidebar"] audio::-webkit-media-controls-play-button:hover {
            background-color: #ff4433 !important;
            transform: scale(1.3) !important;
        }

        /* Webkit Shadow DOM: timeline */
        section[data-testid="stSidebar"] audio::-webkit-media-controls-timeline {
            background-color: rgba(255, 24, 1, 0.15) !important;
            border-radius: 6px !important;
        }

        /* Webkit Shadow DOM: time text */
        section[data-testid="stSidebar"] audio::-webkit-media-controls-current-time-display,
        section[data-testid="stSidebar"] audio::-webkit-media-controls-time-remaining-display {
            color: #8f9cae !important;
            font-family: 'Poppins', monospace !important;
            font-size: 10px !important;
        }

        /* Webkit Shadow DOM: volume */
        section[data-testid="stSidebar"] audio::-webkit-media-controls-volume-slider {
            background-color: rgba(255, 24, 1, 0.2) !important;
            border-radius: 4px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.audio(_audio_bytes, format="audio/mp3", loop=True)

