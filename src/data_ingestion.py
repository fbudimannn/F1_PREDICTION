import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from .utils import CIRCUITS, GRID_2026, CONSTRUCTORS_2026, get_historical_pole_time

# Mapping from internal circuit_id to official FastF1 GP event name (2026 calendar)
CIRCUIT_TO_GP_NAME = {
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
    # Legacy/extra mappings
    "bahrain": "Bahrain Grand Prix",
    "saudi_arabia": "Saudi Arabian Grand Prix",
}

# Official offline calendar database for the 2026 season to guarantee robust, 
# zero-network fallback resolution on hosting servers (e.g. Streamlit Cloud)
OFFICIAL_2026_CALENDAR = {
    "australia": {"date": "2026-03-08", "name": "Australian Grand Prix", "start_utc_hour": 5},
    "china": {"date": "2026-03-15", "name": "Chinese Grand Prix", "start_utc_hour": 7},
    "japan": {"date": "2026-03-29", "name": "Japanese Grand Prix", "start_utc_hour": 5},
    "bahrain": {"date": "2026-04-12", "name": "Bahrain Grand Prix", "start_utc_hour": 15},
    "saudi_arabia": {"date": "2026-04-19", "name": "Saudi Arabian Grand Prix", "start_utc_hour": 17},
    "miami": {"date": "2026-05-03", "name": "Miami Grand Prix", "start_utc_hour": 20},
    "canada": {"date": "2026-05-24", "name": "Canadian Grand Prix", "start_utc_hour": 18},
    "monaco": {"date": "2026-06-07", "name": "Monaco Grand Prix", "start_utc_hour": 13},
    "barcelona": {"date": "2026-06-14", "name": "Barcelona Grand Prix", "start_utc_hour": 13},
    "austria": {"date": "2026-06-28", "name": "Austrian Grand Prix", "start_utc_hour": 13},
    "great_britain": {"date": "2026-07-05", "name": "British Grand Prix", "start_utc_hour": 14},
    "belgium": {"date": "2026-07-19", "name": "Belgian Grand Prix", "start_utc_hour": 13},
    "hungary": {"date": "2026-07-26", "name": "Hungarian Grand Prix", "start_utc_hour": 13},
    "netherlands": {"date": "2026-08-23", "name": "Dutch Grand Prix", "start_utc_hour": 13},
    "italy": {"date": "2026-09-06", "name": "Italian Grand Prix", "start_utc_hour": 13},
    "spain_madrid": {"date": "2026-09-13", "name": "Spanish Grand Prix", "start_utc_hour": 13},
    "azerbaijan": {"date": "2026-09-27", "name": "Azerbaijan Grand Prix", "start_utc_hour": 11},
    "singapore": {"date": "2026-10-11", "name": "Singapore Grand Prix", "start_utc_hour": 12},
    "united_states": {"date": "2026-10-25", "name": "United States Grand Prix", "start_utc_hour": 19},
    "mexico": {"date": "2026-11-01", "name": "Mexico City Grand Prix", "start_utc_hour": 20},
    "brazil": {"date": "2026-11-08", "name": "São Paulo Grand Prix", "start_utc_hour": 17},
    "las_vegas": {"date": "2026-11-21", "name": "Las Vegas Grand Prix", "start_utc_hour": 6},
    "qatar": {"date": "2026-11-29", "name": "Qatar Grand Prix", "start_utc_hour": 17},
    "abu_dhabi": {"date": "2026-12-06", "name": "Abu Dhabi Grand Prix", "start_utc_hour": 13}
}

def get_race_status(circuit_id):
    """
    Determines the status of a Grand Prix for the given circuit:
      - DONE: Race is fully completed (all data available)
      - ONGOING: Race is currently in progress (live lap data streaming)
      - SOON: Race has not started yet (show scheduled date/time)
    
    Uses an offline-first 2026 calendar database to resolve statuses reliably
    independent of network/proxy blocks on hosting servers, with live FastF1
    override attempts.
    """
    gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
    total_laps = CIRCUITS[circuit_id]["laps"]
    
    # 1. Offline-First Calendar Resolution (guarantees robust cloud execution)
    if circuit_id in OFFICIAL_2026_CALENDAR:
        event_info = OFFICIAL_2026_CALENDAR[circuit_id]
        event_date_str = event_info["date"]
        event_name = event_info["name"]
        
        # Parse official race Sunday date (UTC)
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        # Default fallback start time based on specific circuit timezone offsets
        start_hour = event_info.get("start_utc_hour", 14)
        race_start = event_date + timedelta(hours=start_hour)
        
        # Try dynamic lookup from FastF1 schedule to get the exact start time in UTC
        try:
            import fastf1
            fastf1.Cache.enable_cache('fastf1_cache')
            schedule = fastf1.get_event_schedule(2026)
            event_row = schedule[schedule['EventName'].str.lower() == event_name.lower()]
            if len(event_row) == 0:
                first_word = event_name.split(' ')[0].lower()
                event_row = schedule[schedule['EventName'].str.lower().str.contains(first_word, regex=False)]
            
            # If multiple rows matched (e.g. two "Spanish Grand Prix" or similar),
            # pick the one whose EventDate is closest to our offline calendar date
            if len(event_row) > 1:
                event_row['_date_diff'] = (pd.to_datetime(event_row['EventDate']) - pd.Timestamp(event_date_str)).abs()
                event_row = event_row.sort_values('_date_diff').head(1)
            
            if len(event_row) > 0:
                race_start_val = event_row.iloc[0]['Session5DateUtc']
                if pd.notna(race_start_val):
                    candidate_start = None
                    if isinstance(race_start_val, pd.Timestamp):
                        candidate_start = race_start_val.to_pydatetime().replace(tzinfo=timezone.utc)
                    else:
                        candidate_start = pd.to_datetime(race_start_val).to_pydatetime().replace(tzinfo=timezone.utc)
                    
                    # Safety: Only use FastF1's start time if it's within ±3 days of our calendar date
                    # This prevents a wrong GP match from overriding our start time
                    if candidate_start and abs((candidate_start - event_date).total_seconds()) < 3 * 86400:
                        race_start = candidate_start
        except Exception:
            pass
            
        race_end = race_start + timedelta(hours=2, minutes=30)
        now = datetime.now(timezone.utc)
        
        if now > race_end:
            # Race is completed
            try:
                import fastf1
                fastf1.Cache.enable_cache('fastf1_cache')
                
                # Prevent fuzzy match bugs by validating against FastF1 schedule
                schedule = fastf1.get_event_schedule(2026)
                event_row = schedule[schedule['EventName'].str.lower() == event_name.lower()]
                if len(event_row) == 0:
                    first_word = event_name.split(' ')[0].lower()
                    event_row = schedule[schedule['EventName'].str.lower().str.contains(first_word, regex=False)]
                if len(event_row) == 0:
                    raise ValueError(f"GP {event_name} is not in the FastF1 2026 calendar.")
                
                session = fastf1.get_session(2026, event_name, 'R')
                session.load(telemetry=False, weather=False)
                latest_lap = total_laps
                if len(session.laps) > 0:
                    latest_lap = int(session.laps['LapNumber'].max())
                    
                return {
                    "status": "DONE",
                    "event_date": event_date_str,
                    "event_name": event_name,
                    "latest_lap": latest_lap,
                    "total_laps": total_laps
                }
            except Exception as e:
                # If loading fails (no actual data exists in FastF1), mark as SOON (TBD)
                return {
                    "status": "SOON",
                    "event_date": "TBD",
                    "event_name": event_name,
                    "latest_lap": None,
                    "total_laps": total_laps
                }
            
        elif now >= race_start and now <= race_end:
            # Race is currently ongoing/live
            latest_lap = get_latest_available_lap(circuit_id)
            if latest_lap is None:
                # Fallback estimation based on elapsed race time (average ~1.6 mins per lap)
                elapsed_minutes = (now - race_start).total_seconds() / 60.0
                estimated_lap = int(elapsed_minutes / 1.6)
                latest_lap = np.clip(estimated_lap, 1, total_laps)
                
            return {
                "status": "ONGOING",
                "event_date": event_date_str,
                "event_name": event_name,
                "latest_lap": latest_lap,
                "total_laps": total_laps
            }
            
        else:
            # Race is upcoming
            return {
                "status": "SOON",
                "event_date": event_date_str,
                "event_name": event_name,
                "latest_lap": None,
                "total_laps": total_laps
            }

    # 2. Live API fallback if circuit is not in our 2026 database (legacy/extra circuits)
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        
        schedule = fastf1.get_event_schedule(2026)
        event_row = schedule[schedule['EventName'] == gp_name]
        if len(event_row) == 0:
            event_row = schedule[schedule['EventName'].str.contains(gp_name.split(' ')[0], case=False)]
        
        if len(event_row) == 0:
            return {
                "status": "SOON",
                "event_date": "TBD",
                "event_name": gp_name,
                "latest_lap": None,
                "total_laps": total_laps
            }
        
        event_row = event_row.iloc[0]
        event_date = pd.Timestamp(event_row['EventDate'])
        event_name = event_row['EventName']
        
        now = datetime.now(timezone.utc)
        race_start_estimate = event_date.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=14)
        race_end_estimate = race_start_estimate + timedelta(hours=2, minutes=30)
        
        if now > race_end_estimate:
            try:
                session = fastf1.get_session(2026, gp_name, 'R')
                session.load(telemetry=False, weather=False)
                latest_lap = total_laps
                if len(session.laps) > 0:
                    latest_lap = int(session.laps['LapNumber'].max())
                
                return {
                    "status": "DONE",
                    "event_date": str(event_date.date()),
                    "event_name": event_name,
                    "latest_lap": latest_lap,
                    "total_laps": total_laps
                }
            except:
                # If loading fails, return SOON/TBD
                return {
                    "status": "SOON",
                    "event_date": "TBD",
                    "event_name": event_name,
                    "latest_lap": None,
                    "total_laps": total_laps
                }
        elif now >= race_start_estimate and now <= race_end_estimate:
            latest_lap = get_latest_available_lap(circuit_id)
            return {
                "status": "ONGOING",
                "event_date": str(event_date.date()),
                "event_name": event_name,
                "latest_lap": latest_lap,
                "total_laps": total_laps
            }
        else:
            return {
                "status": "SOON",
                "event_date": str(event_date.date()),
                "event_name": event_name,
                "latest_lap": None,
                "total_laps": total_laps
            }
    
    except Exception as e:
        return {
            "status": "SOON",
            "event_date": "TBD",
            "event_name": gp_name,
            "latest_lap": None,
            "total_laps": total_laps
        }


def get_latest_available_lap(circuit_id):
    """
    For ONGOING or DONE races, fetches the latest lap number available 
    from FastF1 race session data. Returns the max LapNumber or None.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        
        gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
        session = fastf1.get_session(2026, gp_name, 'R')
        session.load(telemetry=False, weather=False)
        
        if len(session.laps) > 0:
            return int(session.laps['LapNumber'].max())
    except:
        pass
    
    return None

# Fault-tolerant FastF1 and OpenF1 API Data Ingestor
def fetch_gp_practice_data(circuit_id, session="FP3"):
    """
    Attempts to download completed practice session times using FastF1.
    If FastF1 is unavailable, offline, or still installing, falls back to a 
    highly accurate sircuit-calibrated practice pace generator.
    """
    try:
        # Try to import fastf1 dynamically
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache') # Enable caching
        
        # Load Montreal/Canada or other session
        circuit_meta = CIRCUITS[circuit_id]
        year = 2026 # Grid year
        
        # Resolve circuit_id to official FastF1 GP event name
        gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
        
        # Prevent fuzzy match bugs by validating against FastF1 schedule
        schedule = fastf1.get_event_schedule(year)
        event_row = schedule[schedule['EventName'].str.lower() == gp_name.lower()]
        if len(event_row) == 0:
            first_word = gp_name.split(' ')[0].lower()
            event_row = schedule[schedule['EventName'].str.lower().str.contains(first_word, regex=False)]
        if len(event_row) == 0:
            raise ValueError(f"GP {gp_name} is not in the FastF1 2026 calendar.")
            
        # Download session data with fallback for Sprint format weekends
        actual_session = session
        try:
            ff1_session = fastf1.get_session(year, gp_name, session)
            ff1_session.load(telemetry=False, weather=False)
            actual_session = session
        except Exception as e:
            # If requesting FP3 on a Sprint weekend, try Sprint Qualifying or Practice 1 instead
            if session == "FP3":
                try:
                    ff1_session = fastf1.get_session(year, gp_name, 'Sprint Qualifying')
                    ff1_session.load(telemetry=False, weather=False)
                    actual_session = "Sprint Qualifying"
                except Exception:
                    ff1_session = fastf1.get_session(year, gp_name, 'Practice 1')
                    ff1_session.load(telemetry=False, weather=False)
                    actual_session = "Practice 1"
            else:
                raise e
        
        # Extract lap averages
        fp3_results = {}
        speed_traps = {}
        tyre_codes = {}
        
        for d in GRID_2026.keys():
            # Get driver lap time data
            driver_laps = ff1_session.laps.pick_driver(d)
            if len(driver_laps) > 0:
                # Get median lap time of representative soft/medium runs (exclude in/out laps)
                valid_laps = driver_laps.pick_quicklaps()
                if len(valid_laps) > 0:
                    fp3_results[d] = float(np.median(valid_laps['LapTime'].dt.total_seconds()))
                    speed_traps[d] = float(np.max(valid_laps['SpeedI1'])) # speed trap 1
                    # Extract most used compound compound (Soft: 0, Medium: 1, Hard: 2)
                    most_used_comp = valid_laps['Compound'].value_counts().index[0]
                    tyre_codes[d] = 0 if most_used_comp == "SOFT" else (1 if most_used_comp == "MEDIUM" else 2)
                    continue
            
            # If a specific 2026 driver is not in the older fastf1 cache, trigger local fallback
            raise ImportError("Driver mapping missing in cache")
            
        return fp3_results, speed_traps, tyre_codes, actual_session
        
    except Exception as e:
        # Graceful Fallback Pace Generator (Physically Calibrated to F1 Grid)
        np.random.seed(42)
        circuit_meta = CIRCUITS[circuit_id]
        
        # Dynamic base lap time anchor from historical pole time (with practice delta of +1.8s)
        base_anchor = get_historical_pole_time(circuit_id) + 1.8
        
        fp3_results = {}
        speed_traps = {}
        tyre_codes = {}
        
        for d, info in GRID_2026.items():
            elo = info["base_elo"]
            # Elo influence and constructor car pace offset
            pace_offset = CONSTRUCTORS_2026.get(info["team"], {}).get("pace_offset", 0.0)
            elo_delta = (elo - 1500) * -0.003 + pace_offset
            
            # Tyre selection (Softs are 0.6s faster than Mediums, 1.2s faster than Hards)
            tyre = np.random.choice([0, 1, 2], p=[0.7, 0.25, 0.05]) # 70% use softs in FP3 simulations
            tyre_delta = tyre * 0.6
            
            # Final lap time computation
            noise = np.random.normal(0, 0.08)
            fp3_results[d] = round(base_anchor + elo_delta + tyre_delta + noise, 3)
            
            # Speed traps: derive from circuit type (speed-drag circuits have higher traps)
            circuit_type = circuit_meta.get("type", "balanced-speed")
            if circuit_type == "speed-drag":
                trap_base = 350.0
            elif circuit_type == "downforce-low":
                trap_base = 290.0
            elif circuit_type == "downforce-high":
                trap_base = 320.0
            else:
                trap_base = 335.0
            # Power units: Red Bull/Ferrari engine +10km/h, Sauber/Alpine -8km/h
            engine_power = 6.0 if info["team"] in ["Red Bull", "Ferrari", "McLaren"] else -4.0
            speed_traps[d] = round(trap_base + engine_power + np.random.normal(0, 2.0), 1)
            tyre_codes[d] = int(tyre)
            
        # Determine fallback session name based on event format
        sprint_circuits = {"china", "miami", "austria", "united_states", "brazil", "qatar"}
        if circuit_id in sprint_circuits and session == "FP3":
            actual_session = "Sprint Qualifying (Fallback)"
        else:
            try:
                schedule = fastf1.get_event_schedule(2026)
                event_row = schedule[schedule['EventName'].str.contains(gp_name.split(' ')[0], case=False)]
                if len(event_row) > 0 and event_row.iloc[0]['EventFormat'] == 'sprint_qualifying':
                    actual_session = "Sprint Qualifying (Fallback)"
                else:
                    actual_session = f"{session} (Fallback)"
            except:
                actual_session = f"{session} (Fallback)"
            
        return fp3_results, speed_traps, tyre_codes, actual_session

def fetch_live_session_timing(circuit_id, active_lap=35):
    """
    Fetches mid-race state at a given lap from real FastF1 race data.
    Uses completed 2026 race data to extract actual positions, gaps, tyre compounds,
    tyre life, and pit stop counts at the specified lap.
    Falls back to a simulated live state if no real race data is available.
    """
    # --- ATTEMPT 1: Real race data from FastF1 ---
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        
        gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
        
        # Prevent fuzzy match bugs by validating against FastF1 schedule
        schedule = fastf1.get_event_schedule(2026)
        event_row = schedule[schedule['EventName'].str.lower() == gp_name.lower()]
        if len(event_row) == 0:
            first_word = gp_name.split(' ')[0].lower()
            event_row = schedule[schedule['EventName'].str.lower().str.contains(first_word, regex=False)]
        if len(event_row) == 0:
            raise ValueError(f"GP {gp_name} is not in the FastF1 2026 calendar.")
            
        session = fastf1.get_session(2026, gp_name, 'R')
        session.load(telemetry=False, weather=True)
        
        laps = session.laps
        if len(laps) == 0:
            raise ValueError("No lap data available for this race")
        
        # Get lap data at the target lap
        target_laps = laps[laps['LapNumber'] == active_lap]
        if len(target_laps) == 0:
            # If exact lap not available, find closest available lap
            available_laps = sorted(laps['LapNumber'].unique())
            closest_lap = min(available_laps, key=lambda x: abs(x - active_lap))
            target_laps = laps[laps['LapNumber'] == closest_lap]
            active_lap = closest_lap
        
        # Sort by position at this lap
        target_laps = target_laps.sort_values('Position')
        
        sorted_drivers = []
        gaps = []
        tyre_ages = []
        pit_stops = []
        compounds = []
        leader_time = None
        
        for _, lap in target_laps.iterrows():
            driver = lap['Driver']
            # Only include drivers in our GRID_2026
            if driver not in GRID_2026:
                continue
            
            sorted_drivers.append(driver)
            
            # Calculate gap from leader using cumulative lap times
            if leader_time is None:
                leader_time = lap.get('Time', None)  # Elapsed session time
                gaps.append(0.0)
            else:
                if pd.notna(lap.get('Time', None)) and pd.notna(leader_time):
                    gap_delta = (lap['Time'] - leader_time).total_seconds()
                    gaps.append(round(abs(gap_delta), 2))
                else:
                    # Estimate gap from position
                    gaps.append(round(len(sorted_drivers) * 1.2, 2))
            
            # Tyre life (how many laps on current set)
            tyre_life = lap.get('TyreLife', np.nan)
            if pd.notna(tyre_life):
                tyre_ages.append(int(tyre_life))
            else:
                tyre_ages.append(np.random.randint(8, 20))
            
            # Stint number = approximate pit stops (stint 1 = 0 pits, stint 2 = 1 pit, etc.)
            stint = lap.get('Stint', np.nan)
            if pd.notna(stint):
                pit_stops.append(max(0, int(stint) - 1))
            else:
                pit_stops.append(1)
            
            # Get tyre compound and normalize it
            raw_compound = str(lap.get('Compound', 'MEDIUM')).upper()
            if 'SOFT' in raw_compound:
                normalized_compound = 'Soft'
            elif 'MEDIUM' in raw_compound:
                normalized_compound = 'Medium'
            elif 'HARD' in raw_compound:
                normalized_compound = 'Hard'
            elif 'INTER' in raw_compound:
                normalized_compound = 'Intermediate'
            elif 'WET' in raw_compound:
                normalized_compound = 'Wet'
            else:
                normalized_compound = 'Medium'
            compounds.append(normalized_compound)
        
        # Fill in any missing drivers from GRID_2026 (DNF/not at this lap)
        dnfs = []
        for d in GRID_2026.keys():
            if d not in sorted_drivers:
                sorted_drivers.append(d)
                gaps.append(0.0)
                tyre_ages.append(0)
                pit_stops.append(0)
                compounds.append("Medium")
                dnfs.append(d)
        
        # Extract live weather telemetry closest to target lap time
        track_temp_live = 28.0
        weather_condition_live = "Dry"
        
        try:
            weather = session.weather_data
            if len(weather) > 0 and len(target_laps) > 0:
                target_time = target_laps.iloc[0]['Time']
                time_deltas = (weather['Time'] - target_time).abs()
                closest_idx = time_deltas.idxmin()
                closest_w = weather.loc[closest_idx]
                
                track_temp_live = round(float(closest_w['TrackTemp']), 1)
                rain_live = bool(closest_w['Rainfall'])
                
                if rain_live:
                    leader_lap = target_laps.iloc[0]
                    leader_compound = str(leader_lap.get('Compound', 'MEDIUM')).upper()
                    if 'WET' in leader_compound:
                        weather_condition_live = "Wet / Heavy Rain"
                    elif 'INTER' in leader_compound:
                        weather_condition_live = "Damp / Light Rain"
                    else:
                        weather_condition_live = "Damp / Light Rain"
                else:
                    weather_condition_live = "Dry"
        except Exception:
            pass
        
        if len(sorted_drivers) > 0:
            active_state = {
                "sorted_drivers": sorted_drivers,
                "gaps": gaps,
                "tyre_ages": tyre_ages,
                "pit_stops": pit_stops,
                "compounds": compounds,
                "dnfs": dnfs,
                "track_temp": track_temp_live,
                "weather_condition": weather_condition_live,
                "data_source": f"FastF1 Real Race Data (Lap {active_lap})"
            }
            return active_state
    
    except Exception as e:
        pass  # Fall through to simulation fallback
    
    # --- ATTEMPT 2: Simulated Live State Fallback ---
    np.random.seed(101)
    
    if circuit_id == "canada":
        # Specific realistic order for Montreal GP Lap 35
        sorted_drivers = [
            'ANT', 'VER', 'HAM', 'HAD', 'LEC', 'COL', 'LAW', 'GAS', 'NOR', 'SAI', 
            'BEA', 'PIA', 'HUL', 'BOR', 'OCO', 'PER', 'STR', 'BOT', 'RUS', 'ALO', 'LIN', 'ALB'
        ]
        dnfs = ['RUS', 'ALO', 'LIN', 'ALB']
        
        # Gaps between cars: fully dynamic using uniform random distribution
        gaps = [0.0]
        for i in range(1, len(sorted_drivers)):
            if sorted_drivers[i] in dnfs:
                gaps.append(0.0)
            else:
                gaps.append(round(np.random.uniform(0.5, 3.0), 2))
                
        tyre_ages = []
        pit_stops = []
        compounds = []
        for d in sorted_drivers:
            if d in dnfs:
                tyre_ages.append(0)
                pit_stops.append(0)
                compounds.append("Medium")
            else:
                tyre_ages.append(np.random.randint(4, 12))
                pit_stops.append(1)
                # Assign some variety of dry compounds
                compounds.append(np.random.choice(["Soft", "Medium", "Hard"], p=[0.2, 0.5, 0.3]))
                
        active_state = {
            "sorted_drivers": sorted_drivers,
            "gaps": gaps,
            "tyre_ages": tyre_ages,
            "pit_stops": pit_stops,
            "compounds": compounds,
            "dnfs": dnfs,
            "track_temp": 24.0,
            "weather_condition": "Dry",
            "data_source": "Simulated Fallback (GP Canada Lap 35)"
        }
        return active_state
        
    else:
        drivers = list(GRID_2026.keys())
        # Order drivers based on Elo with some random qualifying shuffles
        sorted_drivers = sorted(drivers, key=lambda x: GRID_2026[x]["base_elo"] + np.random.normal(0, 40), reverse=True)
        
        # Randomly select 1 to 2 DNF drivers from the back
        num_dnfs = np.random.randint(1, 3)
        dnfs = sorted_drivers[-num_dnfs:]
        
        # Gaps between cars
        gaps = [0.0]
        for i in range(1, len(sorted_drivers)):
            if sorted_drivers[i] in dnfs:
                gaps.append(0.0)
            else:
                gaps.append(round(np.random.uniform(0.5, 3.5), 2))
            
        # Tyre age and pit count states
        tyre_ages = []
        pit_stops = []
        compounds = []
        for d in sorted_drivers:
            if d in dnfs:
                tyre_ages.append(0)
                pit_stops.append(0)
                compounds.append("Medium")
            else:
                tyre_ages.append(np.random.randint(8, 22))
                pit_stops.append(1)
                compounds.append(np.random.choice(["Soft", "Medium", "Hard"], p=[0.2, 0.5, 0.3]))
            
        active_state = {
            "sorted_drivers": sorted_drivers,
            "gaps": gaps,
            "tyre_ages": tyre_ages,
            "pit_stops": pit_stops,
            "compounds": compounds,
            "dnfs": dnfs,
            "track_temp": 28.0,
            "weather_condition": "Dry",
            "data_source": "Simulated Fallback"
        }
        return active_state

def fetch_actual_qualifying_results(circuit_id):
    """
    Attempts to fetch actual qualifying results from FastF1 to use as starting grid.
    If FastF1 is unavailable, returns None (so we can fall back to predictions).
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        year = 2026
        # Resolve circuit_id to official FastF1 GP event name
        gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
        
        # Prevent fuzzy match bugs by validating against FastF1 schedule
        schedule = fastf1.get_event_schedule(year)
        event_row = schedule[schedule['EventName'].str.lower() == gp_name.lower()]
        if len(event_row) == 0:
            first_word = gp_name.split(' ')[0].lower()
            event_row = schedule[schedule['EventName'].str.lower().str.contains(first_word, regex=False)]
        if len(event_row) == 0:
            raise ValueError(f"GP {gp_name} is not in the FastF1 2026 calendar.")
            
        # Load Qualifying session
        ff1_session = fastf1.get_session(year, gp_name, 'Q')
        ff1_session.load(laps=False, telemetry=False, weather=False)
        
        # Extract starting grid from classification
        results = ff1_session.results
        if len(results) > 0:
            # Sort by position
            sorted_results = results.sort_values("Position")
            # Extract driver abbreviations
            actual_grid = sorted_results["Abbreviation"].tolist()
            actual_grid = [d for d in actual_grid if d in GRID_2026]
            if len(actual_grid) > 0:
                return actual_grid
    except Exception as e:
        pass
    return None
