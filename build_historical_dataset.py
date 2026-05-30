"""
Historical Dataset Builder for F1 Prediction ML Training.

Collects qualifying and race data from FastF1 API for seasons 2022-2025
and builds a feature matrix suitable for LightGBM training.

Usage:
    python build_historical_dataset.py

Output:
    data/historical/qualifying_2022_2025.csv
    data/historical/race_results_2022_2025.csv
    data/historical/training_features.csv
"""

import sys
import os
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def collect_data(years=None):
    """
    Collects qualifying and race data from FastF1 for the given seasons.
    Returns two DataFrames: qualifying_df, race_df
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]

    import fastf1
    fastf1.Cache.enable_cache('fastf1_cache')

    quali_rows = []
    race_rows = []

    for year in years:
        print(f"\n{'='*60}")
        print(f"Collecting data for {year} season...")
        print(f"{'='*60}")

        try:
            schedule = fastf1.get_event_schedule(year)
            race_events = schedule[schedule['EventFormat'].isin(
                ['conventional', 'sprint_qualifying', 'sprint', 'sprint_shootout']
            )]
        except Exception as e:
            print(f"  Could not load {year} schedule: {e}")
            continue

        for _, event in race_events.iterrows():
            event_name = event['EventName']
            round_num = event['RoundNumber']
            print(f"\n  [{year} R{round_num}] {event_name}")

            # --- Qualifying Data ---
            try:
                q_session = fastf1.get_session(year, event_name, 'Q')
                q_session.load(laps=False, telemetry=False, weather=False, messages=False)

                # Get results for grid positions
                q_results = q_session.results
                if len(q_results) == 0:
                    print(f"    Qualifying: No results data")
                    continue

                # Get pole time
                pole_time = None
                for _, r in q_results.sort_values('Position').iterrows():
                    q3_time = r.get('Q3', pd.NaT)
                    if pd.notna(q3_time):
                        pole_time = q3_time.total_seconds() if hasattr(q3_time, 'total_seconds') else float(q3_time)
                        break

                if pole_time is None or pole_time <= 0:
                    print(f"    Qualifying: No valid pole time found in results")
                    continue

                # Build teammate pairs
                teammates = {}
                for _, r in q_results.iterrows():
                    driver = r.get('Abbreviation', '')
                    team = r.get('TeamName', '')
                    pos = r.get('Position', None)

                    if not driver or not team or pos is None:
                        continue
                    try:
                        pos = int(pos)
                    except (ValueError, TypeError):
                        continue

                    # Get best lap time
                    best_time = None
                    for q_col in ['Q3', 'Q2', 'Q1']:
                        t = r.get(q_col, pd.NaT)
                        if pd.notna(t):
                            try:
                                best_time = t.total_seconds() if hasattr(t, 'total_seconds') else float(t)
                            except Exception:
                                pass
                            if best_time and best_time > 0:
                                break

                    if best_time is None or best_time <= 0:
                        continue

                    gap_to_pole = best_time - pole_time

                    if team not in teammates:
                        teammates[team] = []
                    teammates[team].append({
                        "driver": driver,
                        "team": team,
                        "position": pos,
                        "best_time": best_time,
                        "gap_to_pole": gap_to_pole
                    })

                # Build qualifying rows with teammate gaps
                for team, drivers_list in teammates.items():
                    for d_info in drivers_list:
                        # Find teammate
                        tm_gap = 0.0
                        for other in drivers_list:
                            if other["driver"] != d_info["driver"]:
                                tm_gap = d_info["best_time"] - other["best_time"]
                                break

                        quali_rows.append({
                            "year": year,
                            "round": round_num,
                            "event": event_name,
                            "driver": d_info["driver"],
                            "team": d_info["team"],
                            "quali_position": d_info["position"],
                            "best_quali_time": d_info["best_time"],
                            "gap_to_pole": d_info["gap_to_pole"],
                            "teammate_quali_gap": tm_gap
                        })

                print(f"    Qualifying: {len(teammates)} teams processed")

            except Exception as e:
                print(f"    Qualifying error: {e}")

            # --- Race Data ---
            try:
                r_session = fastf1.get_session(year, event_name, 'R')
                r_session.load(laps=False, telemetry=False, weather=False, messages=False)

                r_results = r_session.results
                if len(r_results) == 0:
                    continue

                # Build teammate pairs for race
                race_teams = {}
                for _, r in r_results.iterrows():
                    driver = r.get('Abbreviation', '')
                    team = r.get('TeamName', '')
                    pos = r.get('Position', None)
                    status = r.get('Status', '')

                    if not driver or not team:
                        continue
                    try:
                        pos = int(pos) if pos is not None else 99
                    except (ValueError, TypeError):
                        pos = 99

                    is_dnf = pos == 99 or (isinstance(status, str) and status not in ['Finished', '+1 Lap', '+2 Laps', '+3 Laps'])
                    grid_pos = r.get('GridPosition', None)
                    try:
                        grid_pos = int(grid_pos) if grid_pos is not None else 20
                    except (ValueError, TypeError):
                        grid_pos = 20

                    if team not in race_teams:
                        race_teams[team] = []
                    race_teams[team].append({
                        "driver": driver,
                        "team": team,
                        "grid_position": grid_pos,
                        "finish_position": pos,
                        "is_dnf": is_dnf,
                        "status": str(status)
                    })

                for team, drivers_list in race_teams.items():
                    for d_info in drivers_list:
                        tm_race_gap = 0.0
                        for other in drivers_list:
                            if other["driver"] != d_info["driver"]:
                                tm_race_gap = d_info["finish_position"] - other["finish_position"]
                                break

                        race_rows.append({
                            "year": year,
                            "round": round_num,
                            "event": event_name,
                            "driver": d_info["driver"],
                            "team": d_info["team"],
                            "grid_position": d_info["grid_position"],
                            "finish_position": d_info["finish_position"],
                            "is_dnf": d_info["is_dnf"],
                            "teammate_race_gap": tm_race_gap,
                            "status": d_info["status"]
                        })

                print(f"    Race: {len(race_teams)} teams processed")

            except Exception as e:
                print(f"    Race error: {e}")

    quali_df = pd.DataFrame(quali_rows)
    race_df = pd.DataFrame(race_rows)

    return quali_df, race_df


def build_feature_matrix(quali_df, race_df):
    """
    Merges qualifying and race data into a unified training feature matrix.
    Features are designed to be TRANSFERABLE across regulation eras.
    """
    if quali_df.empty or race_df.empty:
        print("Warning: Empty data, returning empty feature matrix")
        return pd.DataFrame()

    # Merge qualifying and race on year+round+driver
    merged = pd.merge(
        quali_df,
        race_df[['year', 'round', 'driver', 'grid_position', 'finish_position',
                 'is_dnf', 'teammate_race_gap']],
        on=['year', 'round', 'driver'],
        how='inner',
        suffixes=('_quali', '_race')
    )

    if merged.empty:
        return merged

    # Compute cumulative ELO per driver across the dataset
    # Simple running ELO accumulation
    driver_elos = {}
    elo_column = []

    for _, row in merged.sort_values(['year', 'round']).iterrows():
        driver = row['driver']
        if driver not in driver_elos:
            driver_elos[driver] = 1500.0
        elo_column.append(driver_elos[driver])

        # Update ELO based on qualifying position (simplified)
        # Better position (lower) = higher update
        pos = row.get('quali_position', 10)
        expected_pos = 10.0
        delta = (expected_pos - pos) * 1.5
        driver_elos[driver] = driver_elos[driver] + delta

    merged = merged.sort_values(['year', 'round']).reset_index(drop=True)
    merged['driver_elo_prior'] = elo_column

    # Compute team-level average gap to pole (rolling 3 events)
    team_gaps = merged.groupby(['year', 'round', 'team'])['gap_to_pole'].min().reset_index()
    team_gaps = team_gaps.sort_values(['team', 'year', 'round'])
    team_gaps['constructor_pace_rolling'] = team_gaps.groupby('team')['gap_to_pole'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    # Shift by 1 to make it constructor pace prior (leak-free)
    team_gaps['constructor_pace_prior'] = team_gaps.groupby('team')['constructor_pace_rolling'].shift(1)

    merged = pd.merge(
        merged,
        team_gaps[['year', 'round', 'team', 'constructor_pace_prior']],
        on=['year', 'round', 'team'],
        how='left'
    )

    # Compute driver-level qualifying position history (leak-free)
    merged = merged.sort_values(['year', 'round']).reset_index(drop=True)
    merged['prev_quali_position'] = merged.groupby('driver')['quali_position'].shift(1).fillna(11.0)
    merged['avg_quali_position_last3'] = merged.groupby('driver')['quali_position'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(11.0)

    # Feature matrix columns
    features = pd.DataFrame({
        'year': merged['year'],
        'round': merged['round'],
        'driver': merged['driver'],
        'team': merged['team'],
        'driver_elo_prior': merged['driver_elo_prior'],
        'teammate_quali_gap': merged['teammate_quali_gap'],
        'gap_to_pole': merged['gap_to_pole'],
        'constructor_pace_prior': merged['constructor_pace_prior'].fillna(0.5),
        'prev_quali_position': merged['prev_quali_position'],
        'avg_quali_position_last3': merged['avg_quali_position_last3'],
        'quali_position': merged['quali_position'],
        'grid_position': merged['grid_position'],
        'finish_position': merged['finish_position'],
        'is_dnf': merged['is_dnf'].astype(int),
        'teammate_race_gap': merged['teammate_race_gap'],
        # Target variables for different tasks
        'target_quali_rank': merged['quali_position'],
        'target_race_rank': merged['finish_position']
    })

    return features


def main():
    print("=" * 60)
    print("F1 Historical Dataset Builder")
    print("Collecting data from FastF1 API (2022-2025)")
    print("=" * 60)

    # Collect raw data
    quali_df, race_df = collect_data(years=[2022, 2023, 2024, 2025])

    # Save raw data
    os.makedirs("data/historical", exist_ok=True)

    if not quali_df.empty:
        quali_df.to_csv("data/historical/qualifying_2022_2025.csv", index=False)
        print(f"\nQualifying data: {len(quali_df)} rows saved to data/historical/qualifying_2022_2025.csv")
    else:
        print("\nWarning: No qualifying data collected!")

    if not race_df.empty:
        race_df.to_csv("data/historical/race_results_2022_2025.csv", index=False)
        print(f"Race data: {len(race_df)} rows saved to data/historical/race_results_2022_2025.csv")
    else:
        print("\nWarning: No race data collected!")

    # Build feature matrix
    if not quali_df.empty and not race_df.empty:
        features = build_feature_matrix(quali_df, race_df)
        if not features.empty:
            features.to_csv("data/historical/training_features.csv", index=False)
            print(f"Training features: {len(features)} rows saved to data/historical/training_features.csv")
            print(f"\nFeature columns: {list(features.columns)}")
            print(f"Years covered: {sorted(features['year'].unique())}")
            print(f"Unique drivers: {features['driver'].nunique()}")
            print(f"Unique teams: {features['team'].nunique()}")
        else:
            print("\nWarning: Feature matrix is empty!")

    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
