"""
Season ELO Carryover Engine & Dynamic Constructor Pace Inference.

Accumulates driver ELO ratings across completed 2026 GPs using qualifying (60%)
and race (40%) head-to-head teammate results. Also infers dynamic constructor
pace offsets from actual gap-to-pole data.

Pre-computed results are cached to data/season_elo_cache.json for Streamlit Cloud
compatibility (no re-computation needed on every page load).
"""

import json
import os
import numpy as np
from datetime import datetime, timezone

from .utils import (
    GRID_2026, CONSTRUCTORS_2026, CIRCUITS,
    get_initial_driver_priors, update_elo_ratings
)
from .data_ingestion import OFFICIAL_2026_CALENDAR, CIRCUIT_TO_GP_NAME

# Chronological order of 2026 GP circuit IDs
SEASON_ORDER_2026 = [
    "australia", "china", "japan", "bahrain", "saudi_arabia",
    "miami", "canada", "monaco", "barcelona", "austria",
    "great_britain", "belgium", "hungary", "netherlands", "italy",
    "spain_madrid", "azerbaijan", "singapore", "united_states",
    "mexico", "brazil", "las_vegas", "qatar", "abu_dhabi"
]

# Paths for cache files
CACHE_DIR = "data"
SEASON_ELO_CACHE = os.path.join(CACHE_DIR, "season_elo_cache.json")
CONSTRUCTOR_PACE_CACHE = os.path.join(CACHE_DIR, "constructor_pace_cache.json")
QUALIFYING_HISTORY_CACHE = os.path.join(CACHE_DIR, "qualifying_history_cache.json")


def _get_completed_circuits(up_to_circuit_id=None):
    """
    Returns a list of circuit_ids that are DONE (completed) in chronological order,
    up to but NOT including `up_to_circuit_id`.
    """
    now = datetime.now(timezone.utc)
    completed = []

    for cid in SEASON_ORDER_2026:
        if up_to_circuit_id and cid == up_to_circuit_id:
            break
        if cid in OFFICIAL_2026_CALENDAR:
            event_date_str = OFFICIAL_2026_CALENDAR[cid]["date"]
            from datetime import timedelta
            event_date = datetime.strptime(event_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            race_end = event_date + timedelta(hours=16, minutes=30)
            if now > race_end:
                completed.append(cid)
    return completed


def _validate_gp_exists(gp_name):
    """
    Quick check if a GP exists in FastF1's 2026 schedule.
    Returns True if the session can be created, False otherwise.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        schedule = fastf1.get_event_schedule(2026)
        # Check if any event name matches (case insensitive)
        event_names = schedule['EventName'].str.lower().tolist()
        return gp_name.lower() in event_names
    except Exception:
        return False


def _fetch_qualifying_times(circuit_id):
    """
    Fetches qualifying best lap times per driver from FastF1 for ELO head-to-head.
    Returns dict: {driver_code: best_qualifying_time_seconds} or None on failure.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
        
        # Quick validation to avoid hanging on non-existent sessions
        if not _validate_gp_exists(gp_name):
            return None
        
        session = fastf1.get_session(2026, gp_name, 'Q')
        session.load(laps=True, telemetry=False, weather=False)

        times = {}
        for d in GRID_2026.keys():
            try:
                driver_laps = session.laps.pick_driver(d)
                if len(driver_laps) > 0:
                    valid = driver_laps.pick_quicklaps()
                    if len(valid) > 0:
                        best = float(valid['LapTime'].dt.total_seconds().min())
                        times[d] = best
            except Exception:
                continue
        return times if len(times) >= 10 else None
    except Exception:
        return None


def _fetch_race_positions(circuit_id):
    """
    Fetches race finish positions per driver from FastF1 for ELO head-to-head.
    Returns dict: {driver_code: finish_position (int)} or None on failure.
    Lower position = better (1st = best).
    For ELO, we convert to times-like format where lower = better.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
        
        if not _validate_gp_exists(gp_name):
            return None
        
        session = fastf1.get_session(2026, gp_name, 'R')
        session.load(laps=False, telemetry=False, weather=False)

        results = session.results
        if len(results) == 0:
            return None

        positions = {}
        for _, row in results.iterrows():
            driver = row.get('Abbreviation', '')
            pos = row.get('Position', None)
            if driver in GRID_2026 and pos is not None:
                try:
                    positions[driver] = float(pos)
                except (ValueError, TypeError):
                    continue
        return positions if len(positions) >= 10 else None
    except Exception:
        return None


def _fetch_qualifying_gap_to_pole(circuit_id):
    """
    Fetches the gap (in seconds) of each team's best driver to pole position.
    Used for dynamic constructor pace inference.
    Returns dict: {team_name: gap_to_pole_seconds} or None.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
        
        if not _validate_gp_exists(gp_name):
            return None
        
        session = fastf1.get_session(2026, gp_name, 'Q')
        session.load(laps=True, telemetry=False, weather=False)

        # Get best qualifying time per driver
        driver_best = {}
        for d in GRID_2026.keys():
            try:
                driver_laps = session.laps.pick_driver(d)
                if len(driver_laps) > 0:
                    valid = driver_laps.pick_quicklaps()
                    if len(valid) > 0:
                        driver_best[d] = float(valid['LapTime'].dt.total_seconds().min())
            except Exception:
                continue

        if len(driver_best) < 10:
            return None

        pole_time = min(driver_best.values())

        # Compute best time per team (gap to pole)
        team_gaps = {}
        for d, t in driver_best.items():
            team = GRID_2026[d]["team"]
            gap = t - pole_time
            if team not in team_gaps or gap < team_gaps[team]:
                team_gaps[team] = round(gap, 3)

        return team_gaps
    except Exception:
        return None


def compute_season_elo(up_to_circuit_id=None):
    """
    Computes cumulative Season ELO for all drivers by iterating through
    completed 2026 GPs chronologically.

    For each completed GP:
      - Qualifying head-to-head update (weight 60%)
      - Race finish head-to-head update (weight 40%)
      - Posterior becomes Prior for next GP

    Args:
        up_to_circuit_id: If provided, only accumulate ELOs from GPs
                         that occurred BEFORE this circuit. If None,
                         accumulate from all completed GPs.

    Returns:
        dict: {driver_code: cumulative_elo}
    """
    # Try to load from cache first
    cache = _load_cache(SEASON_ELO_CACHE)
    cache_key = up_to_circuit_id or "__all__"
    if cache and cache_key in cache:
        return cache[cache_key]

    completed = _get_completed_circuits(up_to_circuit_id)
    current_elo = get_initial_driver_priors()

    if len(completed) == 0:
        return current_elo

    for cid in completed:
        # 1. Qualifying ELO update (60% weight)
        quali_times = _fetch_qualifying_times(cid)
        if quali_times:
            current_elo = update_elo_ratings(
                current_elo, quali_times,
                k_factor=24, source_weight=0.6
            )

        # 2. Race ELO update (40% weight)
        # Race positions: lower position = better, same format as times
        race_positions = _fetch_race_positions(cid)
        if race_positions:
            current_elo = update_elo_ratings(
                current_elo, race_positions,
                k_factor=24, source_weight=0.4
            )

    # Save to cache
    _save_to_cache(SEASON_ELO_CACHE, cache_key, current_elo)
    return current_elo


def compute_dynamic_constructor_pace(up_to_circuit_id=None):
    """
    Infers dynamic constructor pace offsets from actual qualifying data
    of completed 2026 GPs. Uses a rolling average of the last 3 GPs
    to detect upgrade/downgrade trends.

    Returns:
        dict: {team_name: {"pace_offset": float, "color": str, "trend": str}}
              pace_offset is in seconds (negative = faster than baseline)
    """
    # Try to load from cache first
    cache = _load_cache(CONSTRUCTOR_PACE_CACHE)
    cache_key = up_to_circuit_id or "__all__"
    if cache and cache_key in cache:
        return cache[cache_key]

    completed = _get_completed_circuits(up_to_circuit_id)

    if len(completed) == 0:
        # No completed GPs, return static constructor data
        return {team: {"pace_offset": data["pace_offset"], "color": data["color"]}
                for team, data in CONSTRUCTORS_2026.items()}

    # Collect gap-to-pole for each completed GP
    all_team_gaps = []  # list of dicts: {team: gap_seconds}
    for cid in completed:
        gaps = _fetch_qualifying_gap_to_pole(cid)
        if gaps:
            all_team_gaps.append(gaps)

    if len(all_team_gaps) == 0:
        return {team: {"pace_offset": data["pace_offset"], "color": data["color"]}
                for team, data in CONSTRUCTORS_2026.items()}

    # Use rolling window of last 3 GPs (or fewer if less available)
    window = all_team_gaps[-3:]

    dynamic_pace = {}
    for team, static_data in CONSTRUCTORS_2026.items():
        gaps_for_team = [gp.get(team, 1.0) for gp in window if team in gp]

        if len(gaps_for_team) > 0:
            avg_gap = float(np.mean(gaps_for_team))
            # Normalize: pole team = most negative offset
            # Scale gap-to-pole into pace_offset range (-0.65 to +0.50)
            # A gap of 0.0s to pole → offset ~ -0.65 (best)
            # A gap of 1.5s to pole → offset ~ +0.50 (worst)
            normalized_offset = round((avg_gap / 1.5) * 1.15 - 0.65, 3)
            normalized_offset = max(-0.65, min(0.50, normalized_offset))

            # Determine trend
            trend = "stable"
            if len(gaps_for_team) >= 2:
                recent = gaps_for_team[-1]
                older = np.mean(gaps_for_team[:-1])
                if recent < older - 0.05:
                    trend = "improving"
                elif recent > older + 0.05:
                    trend = "declining"

            dynamic_pace[team] = {
                "pace_offset": normalized_offset,
                "color": static_data["color"],
                "trend": trend
            }
        else:
            dynamic_pace[team] = {
                "pace_offset": static_data["pace_offset"],
                "color": static_data["color"],
                "trend": "unknown"
            }

    # Save to cache
    _save_to_cache(CONSTRUCTOR_PACE_CACHE, cache_key, dynamic_pace)
    return dynamic_pace


def _load_cache(cache_path):
    """Load JSON cache file. Returns dict or None."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _save_to_cache(cache_path, key, data):
    """Save data to JSON cache file under a key."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache = _load_cache(cache_path) or {}
        cache[key] = data
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _fetch_qualifying_positions(circuit_id):
    """
    Fetches qualifying positions from FastF1 for history calculation.
    Returns dict: {driver_code: position (int)} or None.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('fastf1_cache')
        gp_name = CIRCUIT_TO_GP_NAME.get(circuit_id, circuit_id.replace('_', ' ').title())
        if not _validate_gp_exists(gp_name):
            return None
        session = fastf1.get_session(2026, gp_name, 'Q')
        session.load(laps=False, telemetry=False, weather=False)
        results = session.results
        if len(results) == 0:
            return None
        positions = {}
        for _, row in results.iterrows():
            driver = row.get('Abbreviation', '')
            pos = row.get('Position', None)
            if driver in GRID_2026 and pos is not None:
                try:
                    positions[driver] = int(pos)
                except (ValueError, TypeError):
                    continue
        return positions if len(positions) >= 10 else None
    except Exception:
        return None


def compute_qualifying_history(up_to_circuit_id=None):
    """
    Computes prior qualifying history metrics (prev_quali_position, avg_quali_position_last3)
    by iterating through completed 2026 GPs chronologically.
    """
    cache = _load_cache(QUALIFYING_HISTORY_CACHE)
    cache_key = up_to_circuit_id or "__all__"
    if cache and cache_key in cache:
        return cache[cache_key]

    completed = _get_completed_circuits(up_to_circuit_id)
    driver_history = {d: [] for d in GRID_2026.keys()}
    
    for cid in completed:
        positions = _fetch_qualifying_positions(cid)
        if positions:
            for d in GRID_2026.keys():
                driver_history[d].append(positions.get(d, 11.0))
        else:
            for d in GRID_2026.keys():
                driver_history[d].append(11.0)
                
    history_metrics = {}
    for d in GRID_2026.keys():
        history = driver_history[d]
        if len(history) == 0:
            prev_pos = 11.0
            avg_pos_3 = 11.0
        else:
            prev_pos = float(history[-1])
            avg_pos_3 = float(np.mean(history[-3:]))
        
        history_metrics[d] = {
            "prev_quali_position": round(prev_pos, 2),
            "avg_quali_position_last3": round(avg_pos_3, 2)
        }
        
    _save_to_cache(QUALIFYING_HISTORY_CACHE, cache_key, history_metrics)
    return history_metrics


def precompute_all_season_caches():
    """
    Pre-computes and caches Season ELO, Dynamic Constructor Pace, and Qualifying History
    for ALL circuits. Run this locally before deploying to Streamlit Cloud
    so that the cache files are available without FastF1 API calls.
    """
    print("Pre-computing Season ELO, Constructor Pace, and Qualifying History caches...")
    for cid in SEASON_ORDER_2026:
        print(f"  Processing: {cid}...")
        try:
            elo = compute_season_elo(cid)
            pace = compute_dynamic_constructor_pace(cid)
            quali_hist = compute_qualifying_history(cid)
            print(f"    ELO range: {min(elo.values())}-{max(elo.values())}")
        except Exception as e:
            print(f"    Error: {e}")

    # Also compute the "__all__" key
    elo_all = compute_season_elo(None)
    pace_all = compute_dynamic_constructor_pace(None)
    quali_hist_all = compute_qualifying_history(None)
    print(f"\nAll-season ELO range: {min(elo_all.values())}-{max(elo_all.values())}")
    print("Cache pre-computation complete!")


if __name__ == "__main__":
    precompute_all_season_caches()
