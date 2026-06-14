import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import time
import base64
import importlib

# Force reload helper modules to prevent Streamlit Cloud from caching old files in memory
import src.utils
import src.data_ingestion
import src.models
import src.season_elo

importlib.reload(src.utils)
importlib.reload(src.data_ingestion)
importlib.reload(src.models)
importlib.reload(src.season_elo)

from src.utils import GRID_2026, CIRCUITS, get_initial_driver_priors, update_elo_ratings, format_lap_time, get_driver_color
from src.data_ingestion import fetch_gp_practice_data, fetch_live_session_timing, fetch_actual_qualifying_results, get_race_status, OFFICIAL_2026_CALENDAR
from src.models import QualifyingModel, MonteCarloSimulator
from src.season_elo import compute_season_elo, compute_dynamic_constructor_pace

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
            # Skip character-level fallback if the token looks like a special label
            if t in ["only", "no", "pit", "stint", "zero"]:
                continue
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
    
    /* Circuit Layout SVG Glow & Hover Effect */
    .circuit-svg {
        transition: all 0.3s ease-in-out !important;
    }
    .circuit-svg:hover {
        filter: drop-shadow(0 0 14px rgba(255, 24, 1, 0.75)) !important;
        transform: scale(1.05);
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

# Dynamic custom multiselect tag colors (matching constructor colors)
css_rules = []
for code in GRID_2026.keys():
    color = get_driver_color(code)
    hex_str = color.lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    text_color = "#0d0f12" if brightness > 150 else "#f0f3f6"
    css_rules.append(f"""
    span[data-baseweb="tag"]:has(span[title^="{code} "]) {{
        background-color: {color} !important;
        border-color: {color} !important;
    }}
    span[data-baseweb="tag"]:has(span[title^="{code} "]) span {{
        color: {text_color} !important;
    }}
    span[data-baseweb="tag"]:has(span[title^="{code} "]) svg {{
        fill: {text_color} !important;
        color: {text_color} !important;
    }}
    """)
st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)



# 3. Sidebar Configuration
if os.path.exists("logo f1.png"):
    st.sidebar.image("logo f1.png", use_container_width=True)
st.sidebar.markdown(f"<h1 class='neon-text-red' style='font-size: 28px; margin-top: 10px; margin-bottom: 5px;'>F1 PREDICTION LAB</h1>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p style='font-size: 13px; color: #8f9cae; margin-top: 0px; margin-bottom: 20px; line-height: 1.4;'>Interactive playground to predict driver probabilities dynamically.</p>", unsafe_allow_html=True)

# Circuit Selection
active_circuit = st.sidebar.selectbox(
    "Select Active Circuit:",
    options=list(CIRCUITS.keys()),
    format_func=lambda x: CIRCUITS[x]["name"]
)
circuit_data = CIRCUITS[active_circuit]

# Track circuit changes to reset simulation mode widget state
if "prev_circuit" not in st.session_state:
    st.session_state.prev_circuit = active_circuit
elif st.session_state.prev_circuit != active_circuit:
    if "sim_mode_sidebar" in st.session_state:
        del st.session_state["sim_mode_sidebar"]
    if "mid_lap_sidebar" in st.session_state:
        del st.session_state["mid_lap_sidebar"]
    st.session_state.prev_circuit = active_circuit

# Fetch race status for active circuit (DONE / ONGOING / SOON)
# Always fetch fresh status (no caching) to ensure correct mode detection
# This is critical: the top-level race_status drives ALL downstream behavior
# (simulation mode, weather, monte carlo, tyre display)
@st.cache_data(ttl=10)
def get_cached_race_status(circuit_id):
    return get_race_status(circuit_id)

race_status = get_cached_race_status(active_circuit)

# Initialize session state variables for real-time live refresh
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 10

if "last_seen_lap" not in st.session_state:
    st.session_state.last_seen_lap = {}

if "last_seen_status" not in st.session_state:
    st.session_state.last_seen_status = {}

# Determine if we should auto-poll (race day detection)
# Poll on race day even when SOON so we can detect SOON→ONGOING transition
_is_race_day = False
if active_circuit in OFFICIAL_2026_CALENDAR:
    from datetime import date as _date_cls
    _cal_date_str = OFFICIAL_2026_CALENDAR[active_circuit]["date"]
    _is_race_day = _cal_date_str == str(_date_cls.today())

if race_status["status"] == "ONGOING":
    refresh_time = st.session_state.refresh_interval
elif _is_race_day:
    # On race day, poll every 30s to detect SOON→ONGOING transition
    refresh_time = 30
else:
    refresh_time = None

# Display race status badge in sidebar (Dynamic Auto-Refresh via Fragment)
@st.fragment(run_every=refresh_time)
def render_live_status_sidebar():
    # ALWAYS bypass cache to get the freshest status possible
    # This is essential for detecting SOON→ONGOING and ONGOING→DONE transitions
    try:
        current_status = get_race_status(active_circuit)
    except Exception:
        current_status = race_status  # fallback to cached
        
    status_type = current_status.get("status", "SOON")
    new_lap = current_status.get("latest_lap")
    
    st.markdown("---")
    if status_type == "DONE":
        st.markdown(f"<span class='badge-done'>✅ RACE COMPLETED</span>", unsafe_allow_html=True)
        st.markdown(f"<small style='color: #8f9cae;'>Finished on {current_status['event_date']} — Full data available ({new_lap}/{current_status['total_laps']} laps)</small>", unsafe_allow_html=True)
    elif status_type == "ONGOING":
        st.markdown(f"<span class='badge-ongoing'>🔴 LIVE — LAP {new_lap}/{current_status['total_laps']}</span>", unsafe_allow_html=True)
        
        # Add refresh rate control inside the fragment so the user can change it
        col_ref1, col_ref2 = st.columns([2, 1])
        with col_ref1:
            options = {"5s": 5, "10s": 10, "30s": 30, "60s": 60, "Manual": 999999}
            curr_val = st.session_state.refresh_interval
            default_idx = list(options.values()).index(curr_val) if curr_val in options.values() else 1
            selected_ref = st.selectbox(
                "Refresh Rate:",
                options=list(options.keys()),
                index=default_idx,
                key="live_refresh_rate_select",
                label_visibility="collapsed"
            )
            new_interval = options[selected_ref]
        with col_ref2:
            if st.button("🔄 Now", help="Force refresh now", use_container_width=True):
                if "live_timing_cache" in st.session_state:
                    st.session_state.live_timing_cache.clear()
                st.cache_data.clear()
                st.rerun()
                
        if new_interval != st.session_state.refresh_interval:
            st.session_state.refresh_interval = new_interval
            st.rerun()
            
        if new_interval == 999999:
            st.markdown(f"<small style='color: #ff5252;'>Auto-refresh paused. Click 'Now' to sync.</small>", unsafe_allow_html=True)
        else:
            st.markdown(f"<small style='color: #ff5252;'>Race in progress — Auto-syncing every {selected_ref}</small>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge-soon'>📅 UPCOMING</span>", unsafe_allow_html=True)
        st.markdown(f"<small style='color: #ffd740;'>Race Day: {current_status['event_date']}</small>", unsafe_allow_html=True)
    
    # === STATUS TRANSITION DETECTION ===
    # Detect SOON→ONGOING or ONGOING→DONE transitions and trigger full page rerun
    # This is CRITICAL: a full rerun updates the top-level race_status which
    # controls simulation mode, weather, tyre display, and monte carlo
    prev_status = st.session_state.last_seen_status.get(active_circuit)
    if prev_status is not None and prev_status != status_type:
        st.session_state.last_seen_status[active_circuit] = status_type
        if "live_timing_cache" in st.session_state:
            st.session_state.live_timing_cache.clear()
        st.cache_data.clear()
        if "sim_mode_sidebar" in st.session_state:
            del st.session_state["sim_mode_sidebar"]
        if "mid_lap_sidebar" in st.session_state:
            del st.session_state["mid_lap_sidebar"]
        if status_type == "ONGOING":
            st.toast(f"🟢 Race has started! Switching to LIVE mode...", icon="🏎️")
        elif status_type == "DONE":
            st.toast(f"🏁 Race finished! Full data now available.", icon="🏆")
        time.sleep(1)
        st.rerun()
    else:
        st.session_state.last_seen_status[active_circuit] = status_type
        
    # === LAP CHANGE DETECTION (ONGOING only) ===
    # Trigger full rerun when a new lap is detected to update live timing data
    if status_type == "ONGOING" and new_lap is not None:
        last_seen = st.session_state.last_seen_lap.get(active_circuit)
        if last_seen is None:
            st.session_state.last_seen_lap[active_circuit] = new_lap
        elif new_lap != last_seen:
            st.session_state.last_seen_lap[active_circuit] = new_lap
            if "live_timing_cache" in st.session_state:
                st.session_state.live_timing_cache.clear()
            st.cache_data.clear()
            st.toast(f"🟢 New Lap Detected: Lap {new_lap}! Updating standings...", icon="🏎️")
            time.sleep(1)
            st.rerun()

# Call the fragment inside the sidebar context
with st.sidebar:
    render_live_status_sidebar()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Simulation Mode")

# Default lap target based on status
if race_status["status"] == "ONGOING" and race_status["latest_lap"] is not None:
    default_mid_lap = race_status["latest_lap"]
elif race_status["status"] == "DONE" and race_status["latest_lap"] is not None:
    default_mid_lap = circuit_data["laps"] // 2  # Default to midpoint for replay
else:
    default_mid_lap = circuit_data["laps"] // 2

# Resolve active mode and mid-lap target
sim_mode = "Standard Pre-Race Predictor"
mid_lap = default_mid_lap

# Guard against StreamlitAPIException: ensure session state has one of the active options
if race_status["status"] == "ONGOING":
    radio_options = ["🔴 Live Race Tracking", "Standard Pre-Race Predictor"]
elif race_status["status"] == "DONE":
    radio_options = ["Standard Pre-Race Predictor", "Replay Mid-Race State"]
else:
    radio_options = []

if "sim_mode_sidebar" in st.session_state and st.session_state["sim_mode_sidebar"] not in radio_options:
    del st.session_state["sim_mode_sidebar"]

if race_status["status"] == "ONGOING":
    sim_mode = st.sidebar.radio(
        "Select Mode:", 
        radio_options,
        help="Standard Pre-Race Predictor simulates the race from the start line (Lap 0). Live Race Tracking simulates the remainder of the race from that exact lap forward.",
        key="sim_mode_sidebar"
    )
    if sim_mode == "🔴 Live Race Tracking":
        st.sidebar.caption(f"Currently tracking Live Race at **Lap {default_mid_lap}**")
elif race_status["status"] == "DONE":
    sim_mode = st.sidebar.radio(
        "Select Mode:", 
        radio_options,
        help="Standard Pre-Race Predictor simulates the race from the start line (Lap 0). Replay Mid-Race State allows you to select any completed lap via slider and run simulations from that lap forward.",
        key="sim_mode_sidebar"
    )
    if sim_mode == "Replay Mid-Race State":
        mid_lap = st.sidebar.slider(
            "Select Lap to Replay:",
            min_value=1,
            max_value=race_status["latest_lap"],
            value=default_mid_lap,
            step=1,
            help="Pick which lap to pause the race and run simulations from.",
            key="mid_lap_sidebar"
        )
else:
    st.sidebar.info("🔒 Live tracking will be available once the race starts.")
    sim_mode = "Standard Pre-Race Predictor"

# Track live mode state and fetch live telemetry early
is_live_mode = sim_mode != "Standard Pre-Race Predictor"
active_state = None
if is_live_mode:
    # Session-level cache to avoid repeating slow API/cache reads on every widget click/rerun.
    # Resets completely when browser is refreshed.
    if "live_timing_cache" not in st.session_state:
        st.session_state.live_timing_cache = {}
    
    cache_key = f"{active_circuit}_{mid_lap}"
    if cache_key in st.session_state.live_timing_cache:
        active_state = st.session_state.live_timing_cache[cache_key]
    else:
        active_state = fetch_live_session_timing(active_circuit, active_lap=mid_lap)
        st.session_state.live_timing_cache[cache_key] = active_state

# Apply live telemetry overrides if present in session state
if is_live_mode and active_state is not None:
    if "live_overrides" not in st.session_state:
        st.session_state.live_overrides = {}
    if active_circuit in st.session_state.live_overrides and st.session_state.live_overrides[active_circuit]:
        active_state = active_state.copy()
        active_state["compounds"] = list(active_state["compounds"])
        active_state["tyre_ages"] = list(active_state["tyre_ages"])
        active_state["pit_stops"] = list(active_state["pit_stops"])
        
        overrides = st.session_state.live_overrides[active_circuit]
        for d, ov in overrides.items():
            if d in active_state.get("sorted_drivers", []):
                d_idx = active_state["sorted_drivers"].index(d)
                if "compound" in ov:
                    active_state["compounds"][d_idx] = ov["compound"]
                if "age" in ov:
                    active_state["tyre_ages"][d_idx] = ov["age"]
                if "pit_stops" in ov:
                    active_state["pit_stops"][d_idx] = ov["pit_stops"]

# Auto-Sync Strategy Dropdowns with Current Tyre on mode/lap changes
current_sync_key = f"{active_circuit}_{mid_lap}" if is_live_mode else "pre-race"
if "last_synced_live_state" not in st.session_state:
    st.session_state.last_synced_live_state = None

if st.session_state.last_synced_live_state != current_sync_key:
    if is_live_mode and active_state is not None and "compounds" in active_state:
        active_drivers = active_state.get("sorted_drivers", [])
        for idx, d in enumerate(active_drivers):
            if d in GRID_2026.keys():
                is_dnf = d in active_state.get("dnfs", [])
                if not is_dnf:
                    curr_compound = active_state["compounds"][idx]
                    
                    # Map to a default template that contains/starts with the current compound
                    if curr_compound == "Soft":
                        mapped_strat = "Soft-Hard"
                    elif curr_compound == "Medium":
                        mapped_strat = "Medium-Hard"
                    elif curr_compound == "Hard":
                        mapped_strat = "Hard Only"
                    elif curr_compound == "Intermediate":
                        mapped_strat = "Intermediate-Intermediate"
                    elif curr_compound == "Wet":
                        mapped_strat = "Wet-Intermediate"
                    else:
                        mapped_strat = "Medium-Hard"
                        
                    st.session_state[f"strat_select_{d}"] = mapped_strat
    st.session_state.last_synced_live_state = current_sync_key


st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📍 Circuit Specifications")
st.sidebar.markdown(f"**Laps:** {circuit_data['laps']} | **Length:** {circuit_data['length_km']} km")
st.sidebar.markdown(f"**Safety Car Rate:** {int(circuit_data['sc_probability']*100)}%")
st.sidebar.markdown(f"**Overtaking:** {'Easy/Moderate' if circuit_data['overtaking_index'] > 0.5 else 'Very Hard'}")

# Circuit Layout SVG Display mapping and rendering (glowing white outline)
CIRCUIT_SVG_MAP = {
    "australia": "melbourne-2",
    "china": "shanghai-1",
    "japan": "suzuka-2",
    "bahrain": "bahrain-1",
    "saudi_arabia": "jeddah-1",
    "miami": "miami-1",
    "canada": "montreal-6",
    "monaco": "monaco-6",
    "barcelona": "catalunya-6",
    "austria": "spielberg-3",
    "great_britain": "silverstone-8",
    "belgium": "spa-francorchamps-4",
    "hungary": "hungaroring-3",
    "netherlands": "zandvoort-5",
    "italy": "monza-7",
    "spain_madrid": "madring-1",
    "azerbaijan": "baku-1",
    "singapore": "marina-bay-4",
    "united_states": "austin-1",
    "mexico": "mexico-city-3",
    "brazil": "interlagos-2",
    "las_vegas": "las-vegas-1",
    "qatar": "lusail-1",
    "abu_dhabi": "yas-marina-2",
}

layout_id = CIRCUIT_SVG_MAP.get(active_circuit)
if layout_id:
    svg_url = f"https://raw.githubusercontent.com/julesr0y/f1-circuits-svg/main/circuits/detailed/white-outline/{layout_id}.svg"
    st.sidebar.markdown(f'<div class="circuit-layout-container" style="text-align: center; margin-top: 15px; margin-bottom: 5px;"><img src="{svg_url}" class="circuit-svg" style="max-height: 160px; width: auto; filter: drop-shadow(0 0 8px rgba(255, 24, 1, 0.45));" alt="{circuit_data["name"]} Layout" onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'none\';"><div style="font-size: 10px; color: #8f9cae; margin-top: 6px;">{circuit_data["name"]} Track Layout</div></div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
# Environmental Settings rendering (Conditional)
if is_live_mode and active_state is not None:
    weather_condition = active_state.get("weather_condition", "Dry")
    track_temp = active_state.get("track_temp", 28.0)
    
    # Map weather to rain_intensity
    rain_intensity = 0.0
    if weather_condition == "Damp / Light Rain":
        rain_intensity = 0.5
    elif weather_condition == "Wet / Heavy Rain":
        rain_intensity = 1.0
        
    # Render Locked Weather Card
    weather_icon = "☀️" if "Dry" in weather_condition else ("🌧️" if "Wet" in weather_condition else "☁️")
    st.sidebar.markdown(f"### ☁️ Environmental Settings")
    st.sidebar.markdown(f"""
    <div style="background: rgba(255, 24, 1, 0.05); border: 1px solid rgba(255, 24, 1, 0.25); border-radius: 8px; padding: 12px; margin-bottom: 15px;">
        <div style="font-size: 11px; font-weight: 800; color: #ff1801; letter-spacing: 1px; margin-bottom: 5px;">🟢 LIVE TELEMETRY ACTIVE</div>
        <div style="font-size: 15px; font-weight: 600; color: #f0f3f6; display: flex; align-items: center; gap: 8px;">
            <span>{weather_icon}</span>
            <span>{weather_condition}</span>
        </div>
        <div style="font-size: 13px; color: #8f9cae; margin-top: 4px;">
            🌡️ <strong>Track Temp:</strong> {track_temp} °C
        </div>
        <div style="font-size: 10px; color: #8f9cae; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px; line-height: 1.3;">
            This is the official weather condition recorded at <strong>Lap {mid_lap}</strong> of the race. Manual overrides are disabled.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"### ☁️ Environmental Settings")
    weather_condition = st.sidebar.selectbox(
        "Weather Condition:",
        options=["Dry", "Damp / Light Rain", "Wet / Heavy Rain"],
        index=0,
        key="weather_condition_select"
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

# Auto-Sync Strategy Logic on Weather Change
if "applied_weather" not in st.session_state:
    st.session_state.applied_weather = weather_condition

if st.session_state.applied_weather != weather_condition:
    # Weather changed!
    # Update strategy dropdown selections in session state for all drivers
    if weather_condition == "Wet / Heavy Rain":
        new_strat = "Wet-Intermediate"
    elif weather_condition == "Damp / Light Rain":
        new_strat = "Intermediate-Intermediate"
    else:
        new_strat = "Medium-Hard"
        
    for d in GRID_2026.keys():
        st.session_state[f"strat_select_{d}"] = new_strat
        
    # Save the new weather state
    st.session_state.applied_weather = weather_condition
    
    # Show toast
    st.toast(f"🌧️ Strategies auto-updated for {weather_condition} conditions", icon="⚠️")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Operations")
if st.sidebar.button("Refresh Data & Clear Cache", help="Clears Streamlit cache and forces a full model retrain and data fetch"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
credit_html = """
<div style="font-size: 11px; color: #8f9cae; margin-bottom: 8px;">
    F1 Bayesian Predictor v1.0.0<br>2026 Grid Simulation
</div>
<div style="font-size: 11px; color: #8f9cae; display: flex; align-items: center; gap: 4px; margin-top: 12px;">
    <span>Developed by</span>
    <a href="https://www.linkedin.com/in/muhammad-fakhri-musyaffa-budiman" target="_blank" style="color: #00e5ff; text-decoration: none; font-weight: 600; display: inline-flex; align-items: center;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#0077B5" width="12" height="12" style="margin-right: 4px; vertical-align: middle;">
            <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.779-1.75-1.75s.784-1.75 1.75-1.75 1.75.779 1.75 1.75-.784 1.75-1.75 1.75zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
        </svg>
        Fakhri
    </a>
</div>
"""
render_sidebar_html(credit_html)

# 4. Main Panel Page Title
st.markdown(f"<h1 style='margin-bottom: 5px;'>F1 Bayesian Predictor & Live Tracker <span class='neon-text-red'>2026</span></h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color: #8f9cae; font-size: 16px; margin-bottom: 25px;'>Actively tracking and simulating: <strong>GP {circuit_data['name']}</strong></p>", unsafe_allow_html=True)

# 5. Core Model & Session State Initialization
@st.cache_resource
def load_qualy_model_v2():
    model = QualifyingModel()
    model.load_trained_models()
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

# Initialize driver priors (Elo) using Season ELO Carryover
# Accumulates ELO from all completed GPs before the active circuit
@st.cache_data(ttl=300)
def get_cached_season_elo(circuit_id):
    return compute_season_elo(circuit_id)

@st.cache_data(ttl=300)
def get_cached_constructor_pace(circuit_id):
    return compute_dynamic_constructor_pace(circuit_id)

season_elo = get_cached_season_elo(active_circuit)
dynamic_constructor_pace = get_cached_constructor_pace(active_circuit)

if "driver_priors" not in st.session_state:
    st.session_state.driver_priors = season_elo

# Dynamic Bayesian Elo Rating Update based on practice head-to-heads
# Uses Season ELO (kumulatif) as the base prior instead of static base_elo
st.session_state.driver_priors = update_elo_ratings(season_elo, fp3_times)

# Calculate qualifying results globally so they are always available and persistent across tab switches
qualy_results = qualy_model.predict_qualifying(
    st.session_state.driver_priors, fp3_times, track_temp, speed_traps, tyre_codes,
    rain_intensity=rain_intensity, constructor_pace_dynamic=dynamic_constructor_pace,
    active_circuit=active_circuit
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
        
        y_min = float(df_qualy["best_case_time"].min() - 0.5)
        y_max = float(df_qualy["worst_case_time"].max() + 0.5)
        
        fig_q.update_layout(
            title="Estimated Q3 Lap Times with 90% Bayesian Credible Intervals",
            xaxis_title="Driver Name",
            yaxis=dict(
                range=[y_min, y_max],
                title="Lap Time (seconds)"
            ),
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
    <div style="position: absolute; right: -12px; bottom: 0px; width: 130px; height: 160px; background-image: url('{driver_img}'); background-size: cover; background-position: center top; opacity: 0.50; mask-image: linear-gradient(to left, rgba(0,0,0,1) 15%, rgba(0,0,0,0) 80%); -webkit-mask-image: linear-gradient(to left, rgba(0,0,0,1) 15%, rgba(0,0,0,0) 80%); pointer-events: none; z-index: 0;"></div>
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
    curr_lap = 0
    grid_source_used = "ML Prediction"
    
    # Control Options
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
    
    with col_c1:
        with st.container(border=True):
            st.markdown("#### Simulation Controls", help="Adjust the foundational settings for the Monte Carlo simulator, including run counts, pre-race vs mid-race modes, and starting grid order.")
            sim_iterations = st.selectbox(
                "Simulation Runs:", 
                [5000, 10000, 20000], 
                index=1,
                help="The number of randomized race runs. Higher values (e.g. 20,000) improve statistical precision and smooth out noise in win probabilities, but take slightly longer to execute."
            )
            
            # Display current simulation mode details (which are set in the sidebar)
            if is_live_mode:
                st.markdown(f"**Simulation Mode:** `Live / Replay`")
                st.markdown(f"**Target Lap:** `Lap {mid_lap}`")
                grid_source = "Use Live Running Order"
            else:
                st.markdown(f"**Simulation Mode:** `Standard Pre-Race`")
                grid_source = st.radio(
                    "Starting Grid Source:",
                    ["Use Predicted Qualifying Grid (ML)", "Use Actual Saturday Qualifying Classification (API)"],
                    index=0,
                    help="Choose whether to build the race starting order using our LightGBM Machine Learning grid prediction (Bayesian Prior + FP3 Form) or Saturday's official qualifying classification from the FastF1 API."
                )
            
            if is_live_mode and active_state is not None:
                st.markdown("---")
                with st.expander("🔧 Edit Live Telemetry Overrides", expanded=False):
                    st.markdown("<small style='color: #8f9cae;'>Override live compounds, tyre ages, and pit stops if telemetry is delayed/wrong.</small>", unsafe_allow_html=True)
                    
                    if "live_overrides" not in st.session_state:
                        st.session_state.live_overrides = {}
                    if active_circuit not in st.session_state.live_overrides:
                        st.session_state.live_overrides[active_circuit] = {}
                        
                    driver_to_edit = st.selectbox(
                        "Select Driver to Override:",
                        options=active_state.get("sorted_drivers", []),
                        key="override_driver_select"
                    )
                    
                    if driver_to_edit:
                        d_idx = active_state["sorted_drivers"].index(driver_to_edit)
                        curr_c = active_state["compounds"][d_idx]
                        curr_age = active_state["tyre_ages"][d_idx]
                        curr_pits = active_state["pit_stops"][d_idx]
                        
                        saved_overrides = st.session_state.live_overrides[active_circuit].get(driver_to_edit, {})
                        val_c = saved_overrides.get("compound", curr_c)
                        val_age = saved_overrides.get("age", curr_age)
                        val_pits = saved_overrides.get("pit_stops", curr_pits)
                        
                        compound_options = ["Soft", "Medium", "Hard", "Intermediate", "Wet"]
                        def_c_idx = compound_options.index(val_c) if val_c in compound_options else 1
                        
                        override_c = st.selectbox("Current Compound:", compound_options, index=def_c_idx, key="override_compound")
                        override_age = st.number_input("Tyre Age (laps):", min_value=0, max_value=80, value=int(val_age), step=1, key="override_age")
                        override_pits = st.number_input("Pit Stops Count:", min_value=0, max_value=5, value=int(val_pits), step=1, key="override_pit_stops")
                        
                        col_ov1, col_ov2 = st.columns(2)
                        with col_ov1:
                            if st.button("Apply Override", use_container_width=True, type="primary"):
                                st.session_state.live_overrides[active_circuit][driver_to_edit] = {
                                    "compound": override_c,
                                    "age": override_age,
                                    "pit_stops": override_pits
                                }
                                st.session_state.sim_results = None
                                st.toast(f"✅ Applied override for {driver_to_edit}!", icon="🏎️")
                                time.sleep(0.5)
                                st.rerun()
                        with col_ov2:
                            if st.button("Reset Driver", use_container_width=True):
                                if driver_to_edit in st.session_state.live_overrides[active_circuit]:
                                    del st.session_state.live_overrides[active_circuit][driver_to_edit]
                                st.session_state.sim_results = None
                                st.toast(f"🔄 Reset overrides for {driver_to_edit} to live telemetry", icon="ℹ️")
                                time.sleep(0.5)
                                st.rerun()
                                
                    if st.session_state.live_overrides[active_circuit]:
                        st.markdown("---")
                        if st.button("Clear All Overrides", use_container_width=True):
                            st.session_state.live_overrides[active_circuit].clear()
                            st.session_state.sim_results = None
                            st.toast("🔄 All overrides cleared!", icon="ℹ️")
                            time.sleep(0.5)
                            st.rerun()

            
    if is_live_mode and active_state is not None:
        curr_lap = mid_lap
        starting_grid = active_state["sorted_drivers"]
        data_src = active_state.get("data_source", "Unknown")
        grid_source_used = f"Live Lap {mid_lap} ({data_src})"
    elif not is_live_mode and grid_source == "Use Actual Saturday Qualifying Classification (API)":
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
            st.markdown("#### Strategy Configurator", help="Customize tyre strategy profiles (tyre compound and sequence) for every driver. The physics engine models degradation curves, thermal degradation, grip loss, and pit stop overhead dynamically.")
            
            # --- ⚡ BULK APPLY STRATEGY CONTROLLER ---
            st.markdown("##### ⚡ Bulk Apply Strategy")
            col_bulk1, col_bulk2 = st.columns([1, 1])
            with col_bulk1:
                bulk_mode = st.selectbox(
                    "Bulk Mode:",
                    ["No Bulk Action", "Apply to All Drivers", "Apply to Selected Drivers"],
                    key="bulk_mode"
                )
            with col_bulk2:
                bulk_options = [
                    "Medium-Hard", 
                    "Soft-Medium-Medium", 
                    "Medium-Medium-Hard", 
                    "Soft-Hard",
                    "Intermediate-Intermediate",
                    "Wet-Intermediate",
                    "Intermediate-Medium",
                    "Soft-Intermediate-Wet",
                    "Soft Only",
                    "Medium Only",
                    "Hard Only",
                    "Intermediate Only",
                    "Wet Only",
                    "🔧 Custom Strategy..."
                ]
                bulk_strat = st.selectbox(
                    "Strategy to Apply:",
                    bulk_options,
                    key="bulk_strat",
                    disabled=(bulk_mode == "No Bulk Action")
                )
            
            custom_bulk_val = ""
            if bulk_mode != "No Bulk Action" and bulk_strat == "🔧 Custom Strategy...":
                custom_bulk_val = st.text_input(
                    "Enter Custom Bulk Strategy (e.g. S-H-S):",
                    value="Medium-Hard",
                    key="custom_bulk_val",
                    help="Type custom compounds separated by hyphens (e.g. S-H-S). Compounds: S (Soft), M (Medium), H (Hard), I (Intermediate), W (Wet)."
                )
                
            selected_bulk_drivers = []
            if bulk_mode == "Apply to Selected Drivers":
                selected_bulk_drivers = st.multiselect(
                    "Select Target Drivers:",
                    options=starting_grid,
                    format_func=lambda x: f"{x} ({GRID_2026.get(x, {}).get('name', x)})",
                    key="bulk_drivers"
                )
                
            if bulk_mode != "No Bulk Action":
                if st.button("⚡ Apply Bulk Strategy", use_container_width=True):
                    targets = starting_grid if bulk_mode == "Apply to All Drivers" else selected_bulk_drivers
                    if bulk_mode == "Apply to Selected Drivers" and not targets:
                         st.warning("⚠️ Please select at least one driver to apply the strategy.")
                    else:
                         applied_value = custom_bulk_val if bulk_strat == "🔧 Custom Strategy..." else bulk_strat
                         for d in targets:
                             st.session_state[f"strat_select_{d}"] = bulk_strat
                             if bulk_strat == "🔧 Custom Strategy...":
                                 st.session_state[f"strat_custom_{d}"] = applied_value
                                 
                         st.toast(f"⚡ Bulk strategy '{applied_value}' applied successfully!", icon="✅")
                         st.rerun()
                         
            st.markdown("<div style='margin-bottom: 15px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;'></div>", unsafe_allow_html=True)
            
            # --- INDIVIDUAL DRIVER STRATEGIES ---
            driver_strats = {}
            with st.container(height=380, border=False):
                for idx, d in enumerate(starting_grid):
                    driver_name = GRID_2026.get(d, {}).get("name", d)
                    # Default selection based on driver/grid position
                    default_idx = 0
                    if d == "HAM":
                        default_idx = 1
                    elif d == "VER":
                        default_idx = 0
                    else:
                        default_idx = (idx % 3) if idx < 5 else 0 # Default Medium-Hard for lower grid
                    
                    strategy_options = [
                        "Medium-Hard", 
                        "Soft-Medium-Medium", 
                        "Medium-Medium-Hard", 
                        "Soft-Hard",
                        "Intermediate-Intermediate",
                        "Wet-Intermediate",
                        "Intermediate-Medium",
                        "Soft-Intermediate-Wet",
                        "Soft Only",
                        "Medium Only",
                        "Hard Only",
                        "Intermediate Only",
                        "Wet Only",
                        "🔧 Custom Strategy..."
                    ]
                    
                    selected_opt = st.selectbox(
                        f"{d} ({driver_name}) Strategy:", 
                        strategy_options, 
                        index=default_idx,
                        key=f"strat_select_{d}",
                        help=f"Select a tyre strategy compound sequence for {driver_name}."
                    )
                    
                    if selected_opt == "🔧 Custom Strategy...":
                        custom_input = st.text_input(
                            f"Enter custom strategy for {d} (e.g., S-M-M, M-H):", 
                            value="Medium-Hard",
                            key=f"strat_custom_{d}",
                            help="Type custom compounds separated by hyphens (e.g., S-M-M for Soft-Medium-Medium, M-H for Medium-Hard). Compounds: S (Soft), M (Medium), H (Hard), I (Intermediate), W (Wet)."
                        )
                        driver_strats[d] = custom_input
                    else:
                        driver_strats[d] = selected_opt
                    
                    # If in Live/Replay mode, display their current live tyre status
                    if is_live_mode and active_state is not None:
                        active_drivers = active_state.get("sorted_drivers", [])
                        if d in active_drivers:
                            d_idx = active_drivers.index(d)
                            is_dnf = d in active_state.get("dnfs", [])
                            
                            if is_dnf:
                                st.markdown("<div style='margin-top: -8px; margin-bottom: 8px;'><span style='color: #8f9cae; font-size: 12px; font-weight: 600;'>❌ Retired / DNF</span></div>", unsafe_allow_html=True)
                            else:
                                if "compounds" in active_state:
                                    compound = active_state["compounds"][d_idx]
                                    age = active_state["tyre_ages"][d_idx]
                                    c_class = f"badge-{compound.lower()}"
                                    tyre_html = f"<div style='margin-top: -8px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;'><span style='font-size: 11px; color: #8f9cae;'>Current:</span><span class='strategy-badge {c_class}'>{compound}</span><span style='font-size: 11px; color: #8f9cae;'>({age} laps old)</span></div>"
                                    st.markdown(tyre_html, unsafe_allow_html=True)
                                    
                    # Dynamic visual feedback of parsed strategy
                    parsed_compounds = parse_strategy(driver_strats[d])
                    badge_html = ""
                    for compound in parsed_compounds:
                        c_class = f"badge-{compound.lower()}"
                        badge_html += f"<span class='strategy-badge {c_class}'>{compound}</span>"
                    st.markdown(f"<div style='margin-bottom: 12px;'>{badge_html}</div>", unsafe_allow_html=True)
                    
    with col_c3:
        with st.container(border=True):
            st.markdown("#### Circuit Event Injectors", help="Inject unexpected events into the race simulations. This modifies baseline probabilities for Safety Cars and mechanical/crash retirements (DNFs) derived from historical track data.")
            sc_toggle = st.slider(
                "Safety Car Probability Multiplier:", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Scales the likelihood of Safety Car deployments. A multiplier of 1.5x increases the chance of SC incidents by 50% based on historical rates for this circuit."
            )
            dnf_toggle = st.slider(
                "DNF Baseline Multiplier:", 
                min_value=0.5, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Scales the likelihood of crashes or engine failures causing retirements (DNFs). Useful to model chaotic/high-risk race conditions."
            )
        
    # Compile strategies map dynamically
    tyre_strategies = {}
    for d in starting_grid:
        if d in driver_strats:
            tyre_strategies[d] = parse_strategy(driver_strats[d])
        else:
            # Default strategy based on grid position
            tyre_strategies[d] = ["Medium", "Hard"]
        
    # Render starting grid source info badge
    st.info(f"🚦 **Starting Grid Status:** Using **{grid_source_used}** as starting order")

    # State management to prevent automatic rerun of Monte Carlo simulation when widgets change
    if "sim_results" not in st.session_state or st.session_state.get("sim_circuit") != active_circuit:
        st.session_state.sim_results = None
        st.session_state.sim_circuit = active_circuit
        st.session_state.base_sc_prob = CIRCUITS[active_circuit]["sc_probability"]
        st.session_state.base_dnf_prob = CIRCUITS[active_circuit]["base_dnf_probability"]

    # Detect stale inputs (changes in tyres, sliders, or controls that have not been applied)
    is_stale = False
    if st.session_state.sim_results is not None and "last_sim_inputs" in st.session_state:
        last = st.session_state.last_sim_inputs
        if (last["circuit"] != active_circuit or 
            last["grid_source"] != grid_source_used or 
            last["sim_iterations"] != sim_iterations or 
            last["sim_mode"] != sim_mode or 
            last["sc_toggle"] != sc_toggle or 
            last["dnf_toggle"] != dnf_toggle or 
            last["tyre_strategies"] != tyre_strategies or
            last.get("weather_condition") != weather_condition or
            last.get("track_temp") != track_temp):
            is_stale = True

    if is_stale:
        st.warning("⚠️ **Unapplied Changes Detected:** You have modified simulation controls, tyres, or event injectors. Click **Apply Strategies & Run** below to update the simulation results.")

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    apply_btn = st.button("⚡ Apply Strategies & Run Monte Carlo Simulation", use_container_width=True, type="primary")

    # Run if button clicked OR if we have no cached results yet (first load of this circuit)
    should_run = apply_btn or (st.session_state.sim_results is None)

    if should_run:
        # Execute simulation
        sim_meta = CIRCUITS[active_circuit]
        # Reset to base to avoid compounding multiplier across runs
        sim_meta["sc_probability"] = st.session_state.base_sc_prob * sc_toggle
        sim_meta["base_dnf_probability"] = st.session_state.base_dnf_prob * dnf_toggle
        
        sim_engine = MonteCarloSimulator(active_circuit)
        
        with st.spinner(f"Running {sim_iterations:,} vectorized NumPy race runs..."):
            sim_stats = sim_engine.simulate_race(
                starting_grid, tyre_strategies, num_sims=sim_iterations, current_lap=curr_lap,
                active_state=active_state, rain_intensity=rain_intensity,
                constructor_pace_dynamic=dynamic_constructor_pace
            )
            
        # Store in session state
        st.session_state.sim_results = sim_stats
        st.session_state.last_sim_inputs = {
            "circuit": active_circuit,
            "grid_source": grid_source_used,
            "sim_iterations": sim_iterations,
            "sim_mode": sim_mode,
            "sc_toggle": sc_toggle,
            "dnf_toggle": dnf_toggle,
            "tyre_strategies": tyre_strategies.copy(),
            "weather_condition": weather_condition,
            "track_temp": track_temp,
        }
        
        # Persist race simulation results to local data folder
        with open("data/race_simulations.json", "w") as f:
            json.dump(sim_stats, f, indent=4)
    else:
        # Load from session state cache
        sim_stats = st.session_state.sim_results
        
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

# 7. Live Race Auto-Refresh (Handled in sidebar status fragment)
# This is now handled dynamically in the sidebar status badge fragment to prevent redundant loads.
pass

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

