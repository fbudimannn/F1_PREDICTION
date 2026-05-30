import argparse
import sys
import os
import json
import pandas as pd
from src.utils import GRID_2026, CIRCUITS, get_initial_driver_priors, update_elo_ratings, format_lap_time
from src.data_ingestion import fetch_gp_practice_data
from src.models import QualifyingModel, MonteCarloSimulator
from src.season_elo import compute_season_elo, compute_dynamic_constructor_pace

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
        
    parser = argparse.ArgumentParser(description="F1 2026 Bayesian Predictor & Simulator CLI Pipeline")
    parser.add_argument(
        "--circuit",
        type=str,
        default="canada",
        choices=list(CIRCUITS.keys()),
        help="Circuit ID to run the simulation for (e.g., canada, monaco, monza)"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=28.0,
        help="Track temperature in Celsius (default: 28.0)"
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=10000,
        help="Number of vectorized Monte Carlo simulations to run (default: 10000)"
    )
    parser.add_argument(
        "--weather",
        type=str,
        default="dry",
        choices=["dry", "damp", "wet"],
        help="Weather condition of the session: dry, damp, or wet (default: dry)"
    )
    args = parser.parse_args()

    circuit_id = args.circuit
    track_temp = args.temp
    num_sims = args.sims
    weather = args.weather
    
    rain_intensity = 0.0
    if weather == "damp":
        rain_intensity = 0.5
    elif weather == "wet":
        rain_intensity = 1.0

    circuit_meta = CIRCUITS[circuit_id]
    print("=" * 70)
    print(f"🏎️  F1 2026 BAYESIAN PREDICTOR & SIMULATOR PIPELINE")
    print(f"📍 Active Circuit: {circuit_meta['name']} ({circuit_id})")
    print(f"☁️  Weather: {weather.upper()} (Rain Intensity: {rain_intensity})")
    print(f"🌡️  Track Temperature: {track_temp}°C | 🏁 Laps: {circuit_meta['laps']}")
    print(f"📊 Running {num_sims:,} Monte Carlo Simulations")
    print("=" * 70)

    # 1. Data Ingestion
    fp3_times, speed_traps, tyre_codes, loaded_session = fetch_gp_practice_data(circuit_id, session="FP3")
    print(f"\n[STEP 1] Ingesting {loaded_session} data from API...")
    
    # Save ingested data locally
    os.makedirs("data", exist_ok=True)
    practice_export = {
        "circuit": circuit_id,
        "fp3_times": fp3_times,
        "speed_traps": speed_traps,
        "tyre_codes": tyre_codes
    }
    with open("data/fp3_practice_data.json", "w") as f:
        json.dump(practice_export, f, indent=4)
    print(f"✅ {loaded_session} data ingested and saved to 'data/fp3_practice_data.json'")

    # 2. Bayesian Elo Prior Updates (Season ELO Carryover)
    print("\n[STEP 2] Computing Season ELO Carryover (cumulative from completed GPs)...")
    season_elo = compute_season_elo(circuit_id)
    dynamic_pace = compute_dynamic_constructor_pace(circuit_id)
    updated_priors = update_elo_ratings(season_elo, fp3_times)
    print("✅ Season ELO computed and updated with practice head-to-heads.")
    
    # Display ELO changes from base
    base_priors = get_initial_driver_priors()
    elo_changes = {d: updated_priors[d] - base_priors[d] for d in updated_priors}
    top_movers = sorted(elo_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print("\n  Top ELO movers (vs base):")
    for d, delta in top_movers:
        sign = "+" if delta >= 0 else ""
        print(f"    {d}: {sign}{delta} (base: {base_priors[d]} -> season: {updated_priors[d]})")

    # 3. ML Qualifying Grid Prediction (Learning-To-Rank)
    print("\n[STEP 3] Running LightGBM Ranker and Quantile models...")
    qualy_model = QualifyingModel()
    qualy_model.load_trained_models()
    qualy_results = qualy_model.predict_qualifying(
        updated_priors, fp3_times, track_temp, speed_traps, tyre_codes,
        rain_intensity=rain_intensity, constructor_pace_dynamic=dynamic_pace,
        active_circuit=circuit_id
    )
    
    # Save qualifying predictions
    with open("data/qualifying_predictions.json", "w") as f:
        json.dump(qualy_results, f, indent=4)
    print("✅ Qualifying grid prediction complete. Saved to 'data/qualifying_predictions.json'")

    print("\n⏱️  PREDICTED STARTING GRID (Top 10):")
    print("-" * 70)
    print(f"{'Pos':<4} | {'Code':<4} | {'Driver Name':<20} | {'Team':<15} | {'Median Time':<12}")
    print("-" * 70)
    for p in qualy_results[:10]:
        print(f"P{p['predicted_position']:<2}  | {p['driver_code']:<4} | {p['driver_name']:<20} | {p['team']:<15} | {format_lap_time(p['median_time'])}")
    print("-" * 70)

    # 4. Monte Carlo Race Simulation
    print("\n[STEP 4] Running vectorized Monte Carlo Race Simulator...")
    starting_grid = [p["driver_code"] for p in qualy_results]
    
    # Compile default Medium-Hard strategy map
    tyre_strategies = {d: ["Medium", "Hard"] for d in starting_grid}
    
    sim_engine = MonteCarloSimulator(circuit_id)
    sim_stats = sim_engine.simulate_race(
        starting_grid, tyre_strategies, num_sims=num_sims,
        rain_intensity=rain_intensity, constructor_pace_dynamic=dynamic_pace
    )
    
    # Save simulations output
    with open("data/race_simulations.json", "w") as f:
        json.dump(sim_stats, f, indent=4)
    print("✅ Race simulation complete. Saved to 'data/race_simulations.json'")

    df_sim = pd.DataFrame.from_dict(sim_stats, orient='index').reset_index().rename(columns={"index": "driver_code"})
    df_sim_sorted = df_sim.sort_values("win_probability", ascending=False)

    print("\n🏁 SIMULATED RACE FINISH PROBABILITIES (Top 10):")
    print("-" * 70)
    print(f"{'Driver Name':<20} | {'Team':<15} | {'Win %':<8} | {'Podium %':<9} | {'Top 10 %':<9} | {'DNF %':<6}")
    print("-" * 70)
    for _, row in df_sim_sorted.head(10).iterrows():
        print(f"{row['driver_name']:<20} | {row['team']:<15} | {row['win_probability']:<8.2f} | {row['podium_probability']:<9.2f} | {row['top10_probability']:<9.2f} | {row['dnf_probability']:<6.2f}")
    print("-" * 70)
    print("\n🎉 Pipeline execution completed successfully! All data persisted in 'data/' folder.")
    print("=" * 70)

if __name__ == "__main__":
    main()
