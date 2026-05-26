import numpy as np

# 2026 Grid Database & Initial Bayesian Elo Calibration
GRID_2026 = {
    "VER": {"name": "Max Verstappen", "team": "Red Bull", "is_rookie": False, "base_elo": 1900, "junior_bonus": 0},  # Still elite but car is weaker
    "HAD": {"name": "Isack Hadjar", "team": "Red Bull", "is_rookie": True, "base_elo": 1420, "junior_bonus": 30},   # Red Bull 2026 rookie
    "HAM": {"name": "Lewis Hamilton", "team": "Ferrari", "is_rookie": False, "base_elo": 1850, "junior_bonus": 0},
    "LEC": {"name": "Charles Leclerc", "team": "Ferrari", "is_rookie": False, "base_elo": 1870, "junior_bonus": 0},
    "NOR": {"name": "Lando Norris", "team": "McLaren", "is_rookie": False, "base_elo": 1880, "junior_bonus": 0},
    "PIA": {"name": "Oscar Piastri", "team": "McLaren", "is_rookie": False, "base_elo": 1850, "junior_bonus": 0},  # Highly competitive McLaren
    "RUS": {"name": "George Russell", "team": "Mercedes", "is_rookie": False, "base_elo": 1840, "junior_bonus": 0}, # Mercedes pacesetter
    "ANT": {"name": "Kimi Antonelli", "team": "Mercedes", "is_rookie": False, "base_elo": 1860, "junior_bonus": 0},  # On fire! Star driver in a dominant Mercedes
    "ALO": {"name": "Fernando Alonso", "team": "Aston Martin", "is_rookie": False, "base_elo": 1750, "junior_bonus": 0},
    "STR": {"name": "Lance Stroll", "team": "Aston Martin", "is_rookie": False, "base_elo": 1480, "junior_bonus": 0},
    "GAS": {"name": "Pierre Gasly", "team": "Alpine", "is_rookie": False, "base_elo": 1550, "junior_bonus": 0},
    "COL": {"name": "Franco Colapinto", "team": "Alpine", "is_rookie": False, "base_elo": 1580, "junior_bonus": 0}, # Alpine 2026 signing
    "HUL": {"name": "Nico Hulkenberg", "team": "Audi", "is_rookie": False, "base_elo": 1580, "junior_bonus": 0},
    "BOR": {"name": "Gabriel Bortoleto", "team": "Audi", "is_rookie": True, "base_elo": 1400, "junior_bonus": 50}, # Audi 2026 rookie
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
    "Mercedes": {"pace_offset": -0.65},      # Mercedes dominant sasis/engine package
    "McLaren": {"pace_offset": -0.55},       # Consistent close challenger
    "Ferrari": {"pace_offset": -0.50},       # Elite third-force challenger
    "Red Bull": {"pace_offset": -0.15},      # Red Bull is "kureng" / highly inconsistent
    "Williams": {"pace_offset": -0.10},
    "Haas": {"pace_offset": 0.05},
    "Audi": {"pace_offset": 0.15},
    "VCARB": {"pace_offset": 0.25},
    "Aston Martin": {"pace_offset": 0.35},  # Aston Martin struggles package
    "Cadillac": {"pace_offset": 0.45},
    "Alpine": {"pace_offset": 0.50}
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
        "name": "Monte Carlo",
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

def update_elo_ratings(driver_priors, fp3_results, k_factor=32):
    """
    Updates driver priors (Elo) using FP3 relative teammate comparisons (Bayesian Likelihood Equalizer).
    
    fp3_results: dict mapping driver code to average lap time in seconds (e.g., {"HAM": 72.350, "LEC": 72.430})
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
        
        # Check if both set times in FP3
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
            
            # Bayesian update formula: Posterior = Prior + K * (Actual - Expected)
            updated_priors[driver_a] = int(elo_a + k_factor * (actual_a - expected_a))
            updated_priors[driver_b] = int(elo_b + k_factor * (actual_b - expected_b))
            
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
