import numpy as np

# 2026 Grid Database & Initial Bayesian Elo Calibration
GRID_2026 = {
    "VER": {"name": "Max Verstappen", "team": "Red Bull", "is_rookie": False, "base_elo": 1900, "junior_bonus": 0},  # Still elite but car is weaker
    "HAD": {"name": "Isack Hadjar", "team": "Red Bull", "is_rookie": True, "base_elo": 1680, "junior_bonus": 40},   # Red Bull 2026 rookie (mostly Q3, sometimes Q2)
    "HAM": {"name": "Lewis Hamilton", "team": "Ferrari", "is_rookie": False, "base_elo": 1850, "junior_bonus": 0},
    "LEC": {"name": "Charles Leclerc", "team": "Ferrari", "is_rookie": False, "base_elo": 1870, "junior_bonus": 0},
    "NOR": {"name": "Lando Norris", "team": "McLaren", "is_rookie": False, "base_elo": 1880, "junior_bonus": 0},
    "PIA": {"name": "Oscar Piastri", "team": "McLaren", "is_rookie": False, "base_elo": 1850, "junior_bonus": 0},  # Highly competitive McLaren
    "RUS": {"name": "George Russell", "team": "Mercedes", "is_rookie": False, "base_elo": 1840, "junior_bonus": 0}, # Mercedes pacesetter
    "ANT": {"name": "Kimi Antonelli", "team": "Mercedes", "is_rookie": False, "base_elo": 1860, "junior_bonus": 0},  # On fire! Star driver in a dominant Mercedes
    "ALO": {"name": "Fernando Alonso", "team": "Aston Martin", "is_rookie": False, "base_elo": 1580, "junior_bonus": 0}, # Underperforming in Aston Martin
    "STR": {"name": "Lance Stroll", "team": "Aston Martin", "is_rookie": False, "base_elo": 1480, "junior_bonus": 0},
    "GAS": {"name": "Pierre Gasly", "team": "Alpine", "is_rookie": False, "base_elo": 1550, "junior_bonus": 0},
    "COL": {"name": "Franco Colapinto", "team": "Alpine", "is_rookie": False, "base_elo": 1580, "junior_bonus": 0}, # Alpine 2026 signing
    "HUL": {"name": "Nico Hulkenberg", "team": "Audi", "is_rookie": False, "base_elo": 1580, "junior_bonus": 0},
    "BOR": {"name": "Gabriel Bortoleto", "team": "Audi", "is_rookie": True, "base_elo": 1520, "junior_bonus": 50}, # Audi rookie (Q2 contender)
    "LAW": {"name": "Liam Lawson", "team": "VCARB", "is_rookie": False, "base_elo": 1550, "junior_bonus": 0},      # VCARB 2026 driver
    "LIN": {"name": "Arvid Lindblad", "team": "VCARB", "is_rookie": True, "base_elo": 1380, "junior_bonus": 40},   # VCARB 2026 rookie
    "ALB": {"name": "Alex Albon", "team": "Williams", "is_rookie": False, "base_elo": 1620, "junior_bonus": 0},
    "SAI": {"name": "Carlos Sainz", "team": "Williams", "is_rookie": False, "base_elo": 1780, "junior_bonus": 0},
    "OCO": {"name": "Esteban Ocon", "team": "Haas", "is_rookie": False, "base_elo": 1560, "junior_bonus": 0},
    "BEA": {"name": "Oliver Bearman", "team": "Haas", "is_rookie": False, "base_elo": 1500, "junior_bonus": 20},
    "PER": {"name": "Sergio Perez", "team": "Cadillac", "is_rookie": False, "base_elo": 1520, "junior_bonus": 0},   # Cadillac 2026 driver
    "BOT": {"name": "Valtteri Bottas", "team": "Cadillac", "is_rookie": False, "base_elo": 1540, "junior_bonus": 0} # Cadillac 2026 driver
}

# 2026 Constructor Pace Offsets (seconds delta)
# Reflects actual 2026 hierarchy: Mercedes consistently dominant, McLaren/Ferrari close behind, Red Bull inconsistent (kureng / fourth best)
CONSTRUCTORS_2026 = {
    "Mercedes": {"pace_offset": -0.65, "color": "#00D2BE"},      # Mercedes teal
    "McLaren": {"pace_offset": -0.55, "color": "#FF8700"},       # McLaren papaya orange
    "Ferrari": {"pace_offset": -0.50, "color": "#C40000"},       # Ferrari deep scuderia red
    "Red Bull": {"pace_offset": -0.15, "color": "#3671C6"},      # Red Bull blue
    "Williams": {"pace_offset": -0.10, "color": "#005AFF"},      # Williams royal blue
    "Haas": {"pace_offset": 0.05, "color": "#D9D9D9"},           # Haas grey
    "Audi": {"pace_offset": 0.15, "color": "#FF4D00"},           # Audi Lava Orange-Red
    "VCARB": {"pace_offset": 0.25, "color": "#1045E2"},          # VCARB blue
    "Aston Martin": {"pace_offset": 0.35, "color": "#00594F"},   # Aston Martin racing green
    "Cadillac": {"pace_offset": 0.45, "color": "#999999"},       # Cadillac silver/gray
    "Alpine": {"pace_offset": 0.50, "color": "#0090FF"}          # Alpine blue
}


# Static Circuit Database & Profiles for all 24 races of the 2026 Season
CIRCUITS = {
    "australia": {
        "name": "Melbourne (Albert Park)",
        "laps": 58,
        "length_km": 5.278,
        "type": "balanced-speed",
        "overtaking_index": 0.55,
        "sc_probability": 0.50,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.07, "Medium": 0.035, "Hard": 0.015}
    },
    "china": {
        "name": "Shanghai",
        "laps": 56,
        "length_km": 5.451,
        "type": "balanced-speed",
        "overtaking_index": 0.65,
        "sc_probability": 0.40,
        "base_dnf_probability": 0.09,
        "tyre_deg_coefficients": {"Soft": 0.085, "Medium": 0.042, "Hard": 0.018}
    },
    "japan": {
        "name": "Suzuka",
        "laps": 53,
        "length_km": 5.807,
        "type": "downforce-high",
        "overtaking_index": 0.40,
        "sc_probability": 0.45,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.095, "Medium": 0.048, "Hard": 0.022}
    },
    "bahrain": {
        "name": "Sakhir",
        "laps": 57,
        "length_km": 5.412,
        "type": "traction-braking",
        "overtaking_index": 0.70,
        "sc_probability": 0.35,
        "base_dnf_probability": 0.07,
        "tyre_deg_coefficients": {"Soft": 0.09, "Medium": 0.045, "Hard": 0.019}
    },
    "saudi_arabia": {
        "name": "Jeddah Corniche",
        "laps": 50,
        "length_km": 6.174,
        "type": "speed-drag",
        "overtaking_index": 0.60,
        "sc_probability": 0.70,
        "base_dnf_probability": 0.10,
        "tyre_deg_coefficients": {"Soft": 0.065, "Medium": 0.032, "Hard": 0.014}
    },
    "miami": {
        "name": "Miami",
        "laps": 57,
        "length_km": 5.412,
        "type": "balanced-speed",
        "overtaking_index": 0.55,
        "sc_probability": 0.50,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.075, "Medium": 0.038, "Hard": 0.016}
    },
    "canada": {
        "name": "Montreal (Gilles-Villeneuve)",
        "laps": 70,
        "length_km": 4.361,
        "type": "balanced-speed",
        "overtaking_index": 0.65,
        "sc_probability": 0.65,
        "base_dnf_probability": 0.10,
        "tyre_deg_coefficients": {"Soft": 0.08, "Medium": 0.04, "Hard": 0.018}
    },
    "monaco": {
        "name": "Monaco",
        "laps": 78,
        "length_km": 3.337,
        "type": "downforce-low",
        "overtaking_index": 0.05,
        "sc_probability": 0.70,
        "base_dnf_probability": 0.12,
        "tyre_deg_coefficients": {"Soft": 0.04, "Medium": 0.02, "Hard": 0.009}
    },
    "barcelona": {
        "name": "Barcelona-Catalunya",
        "laps": 66,
        "length_km": 4.657,
        "type": "downforce-high",
        "overtaking_index": 0.45,
        "sc_probability": 0.30,
        "base_dnf_probability": 0.07,
        "tyre_deg_coefficients": {"Soft": 0.085, "Medium": 0.042, "Hard": 0.018}
    },
    "austria": {
        "name": "Spielberg (Red Bull Ring)",
        "laps": 71,
        "length_km": 4.318,
        "type": "traction-braking",
        "overtaking_index": 0.70,
        "sc_probability": 0.40,
        "base_dnf_probability": 0.07,
        "tyre_deg_coefficients": {"Soft": 0.07, "Medium": 0.035, "Hard": 0.015}
    },
    "great_britain": {
        "name": "Silverstone",
        "laps": 52,
        "length_km": 5.891,
        "type": "downforce-high",
        "overtaking_index": 0.60,
        "sc_probability": 0.50,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.09, "Medium": 0.045, "Hard": 0.020}
    },
    "belgium": {
        "name": "Spa-Francorchamps",
        "laps": 44,
        "length_km": 7.004,
        "type": "speed-drag",
        "overtaking_index": 0.75,
        "sc_probability": 0.55,
        "base_dnf_probability": 0.09,
        "tyre_deg_coefficients": {"Soft": 0.08, "Medium": 0.04, "Hard": 0.017}
    },
    "hungary": {
        "name": "Budapest (Hungaroring)",
        "laps": 70,
        "length_km": 4.381,
        "type": "downforce-low",
        "overtaking_index": 0.35,
        "sc_probability": 0.35,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.075, "Medium": 0.038, "Hard": 0.016}
    },
    "netherlands": {
        "name": "Zandvoort",
        "laps": 72,
        "length_km": 4.259,
        "type": "downforce-high",
        "overtaking_index": 0.40,
        "sc_probability": 0.45,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.08, "Medium": 0.04, "Hard": 0.018}
    },
    "italy": {
        "name": "Monza",
        "laps": 53,
        "length_km": 5.793,
        "type": "speed-drag",
        "overtaking_index": 0.75,
        "sc_probability": 0.40,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.09, "Medium": 0.045, "Hard": 0.02}
    },
    "spain_madrid": {
        "name": "Madrid GP (Street hybrid)",
        "laps": 68,
        "length_km": 5.470,
        "type": "traction-braking",
        "overtaking_index": 0.50,
        "sc_probability": 0.55,
        "base_dnf_probability": 0.09,
        "tyre_deg_coefficients": {"Soft": 0.08, "Medium": 0.04, "Hard": 0.017}
    },
    "azerbaijan": {
        "name": "Baku City Circuit",
        "laps": 51,
        "length_km": 6.003,
        "type": "speed-drag",
        "overtaking_index": 0.70,
        "sc_probability": 0.75,
        "base_dnf_probability": 0.12,
        "tyre_deg_coefficients": {"Soft": 0.07, "Medium": 0.035, "Hard": 0.015}
    },
    "singapore": {
        "name": "Singapore (Marina Bay)",
        "laps": 62,
        "length_km": 4.940,
        "type": "downforce-low",
        "overtaking_index": 0.25,
        "sc_probability": 0.80,
        "base_dnf_probability": 0.12,
        "tyre_deg_coefficients": {"Soft": 0.05, "Medium": 0.025, "Hard": 0.011}
    },
    "united_states": {
        "name": "Austin (COTA)",
        "laps": 56,
        "length_km": 5.513,
        "type": "balanced-speed",
        "overtaking_index": 0.60,
        "sc_probability": 0.40,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.08, "Medium": 0.04, "Hard": 0.018}
    },
    "mexico": {
        "name": "Mexico City (Hermanos Rodriguez)",
        "laps": 71,
        "length_km": 4.304,
        "type": "speed-drag",
        "overtaking_index": 0.55,
        "sc_probability": 0.40,
        "base_dnf_probability": 0.09,
        "tyre_deg_coefficients": {"Soft": 0.085, "Medium": 0.042, "Hard": 0.019}
    },
    "brazil": {
        "name": "Sao Paulo (Interlagos)",
        "laps": 71,
        "length_km": 4.309,
        "type": "balanced-speed",
        "overtaking_index": 0.65,
        "sc_probability": 0.55,
        "base_dnf_probability": 0.09,
        "tyre_deg_coefficients": {"Soft": 0.075, "Medium": 0.038, "Hard": 0.016}
    },
    "las_vegas": {
        "name": "Las Vegas",
        "laps": 50,
        "length_km": 6.201,
        "type": "speed-drag",
        "overtaking_index": 0.75,
        "sc_probability": 0.50,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.065, "Medium": 0.032, "Hard": 0.014}
    },
    "qatar": {
        "name": "Lusail",
        "laps": 57,
        "length_km": 5.419,
        "type": "downforce-high",
        "overtaking_index": 0.50,
        "sc_probability": 0.40,
        "base_dnf_probability": 0.08,
        "tyre_deg_coefficients": {"Soft": 0.09, "Medium": 0.045, "Hard": 0.020}
    },
    "abu_dhabi": {
        "name": "Abu Dhabi (Yas Marina)",
        "laps": 58,
        "length_km": 5.281,
        "type": "traction-braking",
        "overtaking_index": 0.55,
        "sc_probability": 0.35,
        "base_dnf_probability": 0.07,
        "tyre_deg_coefficients": {"Soft": 0.075, "Medium": 0.038, "Hard": 0.016}
    }
}

def get_initial_driver_priors():
    """
    Initializes driver Elo ratings factoring in their F1 base Elo and rookie junior premiums.
    """
    priors = {}
    for code, info in GRID_2026.items():
        # Baseline = Base Elo + Junior Success Premium
        priors[code] = info["base_elo"] + info["junior_bonus"]
    return priors

def compute_expected_performance(elo_a, elo_b):
    """
    Calculates the expected head-to-head performance ratio between two drivers based on Elo.
    Returns value between 0 and 1 (0.5 means equal chance).
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo_ratings(driver_priors, fp3_results, k_factor=24, source_weight=1.0):
    """
    Updates driver priors (Elo) using relative teammate comparisons (Bayesian Likelihood Equalizer).
    
    fp3_results: dict mapping driver code to performance metric in seconds (lap time, etc.)
                 Lower is better (faster).
    k_factor: Sensitivity of each update. Default 24 for stable season-long accumulation.
    source_weight: Multiplier for the update magnitude (e.g., 0.6 for qualifying, 0.4 for race).
                   Allows weighting different data sources differently in cumulative ELO.
    """
    updated_priors = driver_priors.copy()
    
    # 1. Teammate relative comparisons
    # Find teammates and update their rating relative to each other
    teams = {}
    for code, info in GRID_2026.items():
        team = info["team"]
        if team not in teams:
            teams[team] = []
        teams[team].append(code)
        
    for team, drivers in teams.items():
        if len(drivers) != 2:
            continue
        
        driver_a, driver_b = drivers[0], drivers[1]
        
        # Check if both set times
        if driver_a in fp3_results and driver_b in fp3_results:
            time_a = fp3_results[driver_a]
            time_b = fp3_results[driver_b]
            
            # Outcome: 1 if A is faster than B, 0 if B is faster than A
            actual_a = 1.0 if time_a < time_b else 0.0
            actual_b = 1.0 - actual_a
            
            elo_a = driver_priors[driver_a]
            elo_b = driver_priors[driver_b]
            
            expected_a = compute_expected_performance(elo_a, elo_b)
            expected_b = 1.0 - expected_a
            
            # Bayesian update formula: Posterior = Prior + K * weight * (Actual - Expected)
            effective_k = k_factor * source_weight
            updated_priors[driver_a] = int(elo_a + effective_k * (actual_a - expected_a))
            updated_priors[driver_b] = int(elo_b + effective_k * (actual_b - expected_b))
            
    return updated_priors

def format_lap_time(seconds):
    """
    Formats a lap time in seconds (float) into F1 standard format 'M:SS.ms'.
    Example: 73.582 -> '1:13.582'
             58.120 -> '58.120'
    """
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "N/A"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 0:
        return f"{minutes}:{remaining_seconds:06.3f}"
    else:
        return f"{remaining_seconds:.3f}"

def get_driver_color(driver_code):
    """
    Returns the hex color code for a driver based on their team/constructor.
    """
    driver_info = GRID_2026.get(driver_code)
    if not driver_info:
        return "#8f9cae" # default gray
    team_name = driver_info.get("team")
    team_info = CONSTRUCTORS_2026.get(team_name)
    if not team_info:
        return "#8f9cae"
    return team_info.get("color", "#8f9cae")


# Module-level cache for pole times to avoid repeated FastF1 API calls
_pole_time_cache = {}

def get_historical_pole_time(active_circuit):
    """
    Gets the best available pole lap time reference for the active circuit.
    Results are cached in-memory to avoid repeated FastF1 API calls.
    
    Priority order:
    1. Actual 2026 Qualifying pole time from FastF1 (most accurate)
    2. Actual 2026 Sprint Qualifying pole time from FastF1
    3. Historical 2022-2025 average pole time with 2026 regulation speed correction (~3.5s faster)
    4. Length-based calculation formula (last resort)
    """
    # Check in-memory cache first
    if active_circuit in _pole_time_cache:
        return _pole_time_cache[active_circuit]
    
    import os
    
    # Mapping from circuit_id to GP name for FastF1 lookups
    circuit_to_gp = {
        "australia": "Australian Grand Prix",
        "china": "Chinese Grand Prix",
        "japan": "Japanese Grand Prix",
        "miami": "Miami Grand Prix",
        "canada": "Canadian Grand Prix",
        "monaco": "Monaco Grand Prix",
        "barcelona": "Barcelona Grand Prix",
        "austria": "Austrian Grand Prix",
        "great_britain": "British Grand Prix",
        "belgium": "Belgian Grand Prix",
        "hungary": "Hungarian Grand Prix",
        "netherlands": "Dutch Grand Prix",
        "italy": "Italian Grand Prix",
        "spain_madrid": "Spanish Grand Prix",
        "azerbaijan": "Azerbaijan Grand Prix",
        "singapore": "Singapore Grand Prix",
        "united_states": "United States Grand Prix",
        "mexico": "Mexico City Grand Prix",
        "brazil": "São Paulo Grand Prix",
        "las_vegas": "Las Vegas Grand Prix",
        "qatar": "Qatar Grand Prix",
        "abu_dhabi": "Abu Dhabi Grand Prix",
        "bahrain": "Bahrain Grand Prix",
        "saudi_arabia": "Saudi Arabian Grand Prix",
    }
    gp_name = circuit_to_gp.get(active_circuit)
    
    # Map to historical names in qualifying_2022_2025.csv if different from 2026 schedule names
    circuit_to_historical_gp = {
        "barcelona": "Spanish Grand Prix",
        "spain_madrid": "Spanish Grand Prix",
    }
    hist_gp_name = circuit_to_historical_gp.get(active_circuit, gp_name)
    
    # ── Priority 1 & 2: Actual 2026 data from FastF1 ──
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        
        # Validate event exists in the 2026 schedule
        schedule = fastf1.get_event_schedule(2026)
        if gp_name:
            event_row = schedule[schedule['EventName'].str.lower() == gp_name.lower()]
            if len(event_row) == 0:
                first_word = gp_name.split(' ')[0].lower()
                event_row = schedule[schedule['EventName'].str.lower().str.contains(first_word, regex=False)]
            
            if len(event_row) > 0:
                # Try Qualifying first (most representative), then Sprint Qualifying
                for session_type in ['Q', 'Sprint Qualifying']:
                    try:
                        session = fastf1.get_session(2026, gp_name, session_type)
                        session.load(telemetry=False, weather=False)
                        if len(session.laps) > 0:
                            quick_laps = session.laps.pick_quicklaps()
                            if len(quick_laps) > 0:
                                pole_time = quick_laps['LapTime'].min().total_seconds()
                                if 40.0 < pole_time < 200.0:  # Sanity check
                                    _pole_time_cache[active_circuit] = float(pole_time)
                                    return _pole_time_cache[active_circuit]
                    except Exception:
                        continue
    except Exception:
        pass
    
    # ── Priority 3: Historical 2022-2025 Clean Dry Average ──
    # We filter out wet qualifying sessions (defined as pole times > 3.0s slower than the circuit's minimum)
    # to get a representative dry-weather baseline.
    try:
        import pandas as pd
        
        csv_path = "data/historical/qualifying_2022_2025.csv"
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "historical", "qualifying_2022_2025.csv")
            
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Find the best qualifying time for each event and year
            pole_times = df[df['quali_position'] == 1].groupby(['event', 'year'])['best_quali_time'].min().reset_index()
            
            # Find the minimum pole time across all years to establish the absolute dry pace limit
            min_poles = pole_times.groupby('event')['best_quali_time'].min().to_dict()
            
            if hist_gp_name and hist_gp_name in min_poles:
                event_poles = pole_times[pole_times['event'] == hist_gp_name]
                min_val = min_poles[hist_gp_name]
                
                # Filter out years that are > 3.0 seconds slower than the minimum (wet sessions)
                dry_poles = event_poles[event_poles['best_quali_time'] - min_val <= 3.0]['best_quali_time'].tolist()
                
                if dry_poles:
                    historical_pole = sum(dry_poles) / len(dry_poles)
                else:
                    historical_pole = min_val
                
                _pole_time_cache[active_circuit] = float(historical_pole)
                return _pole_time_cache[active_circuit]
    except Exception as e:
        print(f"[get_historical_pole_time] Warning: {e}")
        
    # ── Priority 4: Length-based formula (last resort, already calibrated for 2026) ──
    meta = CIRCUITS.get(active_circuit, CIRCUITS["canada"])
    length = meta["length_km"]
    ctype = meta.get("type", "balanced-speed")
    # These sec_per_km values are calibrated for 2026 car performance
    if ctype == "speed-drag":
        sec_per_km = 12.8
    elif ctype == "traction-braking":
        sec_per_km = 13.7
    elif ctype == "downforce-high":
        sec_per_km = 14.4
    elif ctype == "downforce-low":
        sec_per_km = 15.9
    else:
        sec_per_km = 13.4
    _pole_time_cache[active_circuit] = float(length * sec_per_km)
    return _pole_time_cache[active_circuit]

