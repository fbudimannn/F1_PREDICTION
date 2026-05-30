import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import joblib
from .utils import CIRCUITS, GRID_2026, CONSTRUCTORS_2026, get_historical_pole_time

class QualifyingModel:
    """
    ML Qualifying Predictor using LightGBM.
    Combines:
    1. LightGBM Ranker (Lambdarank) for final relative grid order (P1-P20).
    2. Quantile Regression for lap time credible intervals.
    """
    MODELS_DIR = "models"
    
    def __init__(self):
        self.ranker = None
        self.quantile_models = {}
        self._using_trained_models = False
    
    def load_trained_models(self):
        """
        Loads pre-trained LightGBM models from disk (joblib format).
        Falls back to train_mock_models() if model files are not found.
        """
        ranker_path = os.path.join(self.MODELS_DIR, "qualifying_ranker.joblib")
        if not os.path.exists(ranker_path):
            print("[QualifyingModel] Trained models not found, falling back to mock models.")
            self.train_mock_models()
            return
        
        try:
            self.ranker = joblib.load(ranker_path)
            
            for q in [0.10, 0.50, 0.90]:
                q_path = os.path.join(self.MODELS_DIR, f"lap_time_q{int(q*100):02d}.joblib")
                if os.path.exists(q_path):
                    self.quantile_models[q] = joblib.load(q_path)
            
            self._using_trained_models = True
            print("[QualifyingModel] Successfully loaded pre-trained models.")
        except Exception as e:
            print(f"[QualifyingModel] Error loading models: {e}. Falling back to mock.")
            self.train_mock_models()
        
    def train_mock_models(self, track_type="balanced-speed"):
        """
        Calibrates model anchors using the updated leak-free feature matrix.
        """
        np.random.seed(42)
        n_samples = 220
        
        # Generate new features
        driver_elo_prior = np.random.normal(1600, 150, n_samples)
        constructor_pace_prior = np.random.uniform(-0.6, 0.5, n_samples)
        prev_quali_position = np.random.uniform(1.0, 20.0, n_samples)
        avg_quali_position_last3 = np.random.uniform(1.0, 20.0, n_samples)
        
        # Target gap to pole
        y_time = (0.5 * constructor_pace_prior - (driver_elo_prior - 1500) * 0.001 
                  + prev_quali_position * 0.05 + avg_quali_position_last3 * 0.05
                  + np.random.normal(0, 0.1, n_samples))
        y_time = np.clip(y_time, 0.0, 5.0)
        
        # Target relative ranks/relevance (20 for fastest, 1 for slowest per group of 20)
        y_rank = np.zeros(n_samples)
        for i in range(0, n_samples, 20):
            times_group = y_time[i:i+20]
            y_rank[i:i+20] = 20 - np.argsort(np.argsort(times_group))
            
        X = pd.DataFrame({
            "driver_elo_prior": driver_elo_prior,
            "constructor_pace_prior": constructor_pace_prior,
            "prev_quali_position": prev_quali_position,
            "avg_quali_position_last3": avg_quali_position_last3
        })
        
        # 1. Fit Lambdarank Model for relative ranking
        group = [20] * (n_samples // 20)
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 3, 5, 10],
            n_estimators=30,
            learning_rate=0.1,
            num_leaves=7,
            verbose=-1
        )
        self.ranker.fit(X, y_rank, group=group)
        
        # 2. Fit Quantile Regressors for lap time confidence intervals
        quantiles = [0.10, 0.50, 0.90] # Best case, Median, Worst case
        for q in quantiles:
            q_model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=30,
                learning_rate=0.1,
                num_leaves=7,
                verbose=-1
            )
            q_model.fit(X, y_time)
            self.quantile_models[q] = q_model

    def predict_qualifying(self, driver_priors, fp3_results, track_temp, speed_traps, tyre_codes, rain_intensity=0.0, constructor_pace_dynamic=None, active_circuit="canada"):
        """
        Predicts final qualifying grid ranking and time intervals with strictly
        rank-consistent and highly realistic 2026 paddock distributions.
        
        driver_priors: dict of active updated Elos
        fp3_results: dict of average FP3 lap times
        track_temp: float
        speed_traps: dict of maximum speeds in FP3
        tyre_codes: dict of active tyre compounds (0: Soft, 1: Medium, 2: Hard, 3: Intermediate, 4: Wet)
        rain_intensity: float (0.0: Dry, 0.5: Damp, 1.0: Wet)
        constructor_pace_dynamic: optional dict of dynamic constructor pace data.
                                  If provided, overrides static CONSTRUCTORS_2026 pace offsets.
                                  Format: {team_name: {"pace_offset": float, "color": str}}
        active_circuit: str - circuit ID (e.g. "canada", "monaco"). Passed explicitly by the caller.
        """
        drivers = list(GRID_2026.keys())

        # Load qualifying history for prior position features
        from .season_elo import compute_qualifying_history
        quali_history = compute_qualifying_history(active_circuit)

        # Build feature DataFrame for all drivers
        features_list = []
        for d in drivers:
            elo = driver_priors.get(d, 1500.0)
            team_name = GRID_2026[d]["team"]
            
            if constructor_pace_dynamic and team_name in constructor_pace_dynamic:
                pace_offset = constructor_pace_dynamic[team_name].get("pace_offset", 0.0)
            else:
                pace_offset = CONSTRUCTORS_2026.get(team_name, {}).get("pace_offset", 0.0)
                
            history = quali_history.get(d, {})
            prev_pos = history.get("prev_quali_position", 11.0)
            avg_pos_3 = history.get("avg_quali_position_last3", 11.0)
            
            features_list.append({
                "driver_elo_prior": elo,
                "constructor_pace_prior": pace_offset,
                "prev_quali_position": prev_pos,
                "avg_quali_position_last3": avg_pos_3
            })
            
        features_df = pd.DataFrame(features_list)

        # 2. Get LightGBM Ranker Score (or fallback if models not loaded)
        if self._using_trained_models and self.ranker is not None:
            # Predict using LightGBM ranker
            lgbm_scores = self.ranker.predict(features_df)
            
            # Normalize lgbm scores to 0-1 range (higher = better)
            min_lgbm = min(lgbm_scores)
            max_lgbm = max(lgbm_scores)
            lgbm_range = (max_lgbm - min_lgbm) if max_lgbm > min_lgbm else 1.0
            lgbm_norm = (lgbm_scores - min_lgbm) / lgbm_range
        else:
            # Fallback to manual scoring (elo + car pace normalized)
            lgbm_norm = []
            for d in drivers:
                elo = driver_priors.get(d, 1500.0)
                team_name = GRID_2026[d]["team"]
                if constructor_pace_dynamic and team_name in constructor_pace_dynamic:
                    pace_offset = constructor_pace_dynamic[team_name].get("pace_offset", 0.0)
                else:
                    pace_offset = CONSTRUCTORS_2026.get(team_name, {}).get("pace_offset", 0.0)
                
                elo_score = np.clip((elo - 1300) / 700.0, 0.0, 1.0)
                car_score = np.clip((-pace_offset + 0.50) / 1.15, 0.0, 1.0)
                fallback_score = (elo_score * 0.50 + car_score * 0.35) / 0.85
                lgbm_norm.append(fallback_score)
            lgbm_norm = np.array(lgbm_norm)

        # 3. Practice weekend form (FP3) normalization
        tyre_deltas = [0.0, 0.6, 1.2, 0.0, 0.0]
        norm_fp3_all = {}
        for d in drivers:
            raw_fp3 = fp3_results.get(d, 75.0)
            t_code = tyre_codes.get(d, 0)
            norm_fp3_all[d] = raw_fp3 - tyre_deltas[np.clip(t_code, 0, 4)]
            
        best_norm_fp3 = min(norm_fp3_all.values()) if len(norm_fp3_all) > 0 else 73.5
        
        fp3_scores = []
        for d in drivers:
            fp3_delta = norm_fp3_all.get(d, best_norm_fp3) - best_norm_fp3
            fp3_score = np.clip(1.0 - (fp3_delta / 3.0), 0.0, 1.0)
            fp3_scores.append(fp3_score)
        fp3_scores = np.array(fp3_scores)

        # 4. Hybrid Score: 85% LightGBM Ranker Score + 15% practice pace score
        final_scores = lgbm_norm * 0.85 + fp3_scores * 0.15
        
        scores = []
        for idx, d in enumerate(drivers):
            # Add a small stochastic qualifying variance (organic session variance)
            np.random.seed(hash(d) % 10000 + int(track_temp * 10))
            variance = np.random.normal(0, 0.025)
            scores.append((d, final_scores[idx] + variance))
            
        # Sort drivers descending by score
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        predicted_grid = [x[0] for x in sorted_scores]

        # 5. Determine realistic baseline qualifying pole lap time for the circuit
        base_pole_time = get_historical_pole_time(active_circuit)
        
        if rain_intensity == 0.5:
            base_pole_time += 7.5
        elif rain_intensity == 1.0:
            base_pole_time += 15.0
            
        base_pole_time += max(0, track_temp - 30) * 0.04

        # 6. Predict lap times (quantile regression with monotonicity)
        predictions = {}
        
        if self._using_trained_models and len(self.quantile_models) == 3:
            # Predict gaps for each driver using the quantile models
            q_gaps = {}
            for q in [0.10, 0.50, 0.90]:
                q_model = self.quantile_models[q]
                q_gaps[q] = q_model.predict(features_df)
                
            # Build dict of predictions for each driver
            for idx, d in enumerate(drivers):
                best_gap = max(0.0, q_gaps[0.10][idx])
                median_gap = max(best_gap, q_gaps[0.50][idx])
                worst_gap = max(median_gap, q_gaps[0.90][idx])
                
                predictions[d] = {
                    "best": best_gap,
                    "median": median_gap,
                    "worst": worst_gap
                }
                
            # Enforce monotonicity along the predicted grid
            current_median = 0.0
            current_best = -0.4
            current_worst = 0.6
            gap_multiplier = 1.0 + rain_intensity * 1.5
            
            for rank, d in enumerate(predicted_grid):
                pred = predictions[d]
                if rank == 0:
                    current_median = pred["median"]
                    current_best = pred["best"]
                    current_worst = pred["worst"]
                else:
                    min_step = 0.03 * gap_multiplier
                    current_median = max(current_median + min_step, pred["median"])
                    current_best = max(current_best + min_step, pred["best"])
                    current_worst = max(current_worst + min_step, pred["worst"])
                    
                predictions[d] = {
                    "best": current_best,
                    "median": current_median,
                    "worst": current_worst
                }
        else:
            # Fallback to dynamic Bayesian gaps if models not loaded
            gap_multiplier = 1.0 + rain_intensity * 1.5
            current_time = base_pole_time
            
            for rank, d in enumerate(predicted_grid):
                if rank == 0:
                    predictions[d] = {
                        "median": current_time,
                        "best": current_time - (0.4 * gap_multiplier),
                        "worst": current_time + (0.6 * gap_multiplier)
                    }
                else:
                    base_gap = 0.12 / (1.0 + rank * 0.04)
                    np.random.seed(hash(d) % 5000 + rank)
                    actual_gap = base_gap * np.random.uniform(0.7, 1.3) * gap_multiplier
                    
                    current_time += actual_gap
                    
                    is_rookie = GRID_2026[d]["is_rookie"]
                    interval_width = (1.0 if not is_rookie else 1.45) * gap_multiplier
                    
                    predictions[d] = {
                        "median": current_time,
                        "best": current_time - (0.4 * interval_width),
                        "worst": current_time + (0.6 * interval_width)
                    }

        # 7. Compile final predictions structure sorted by predicted grid rank
        sorted_predictions = []
        for rank, d in enumerate(predicted_grid):
            p = predictions[d]
            if self._using_trained_models and len(self.quantile_models) == 3:
                best_time = base_pole_time + p["best"]
                median_time = base_pole_time + p["median"]
                worst_time = base_pole_time + p["worst"]
            else:
                best_time = p["best"]
                median_time = p["median"]
                worst_time = p["worst"]
                
            sorted_predictions.append({
                "predicted_position": rank + 1,
                "driver_code": d,
                "driver_name": GRID_2026[d]["name"],
                "team": GRID_2026[d]["team"],
                "best_case_time": round(float(best_time), 3),
                "median_time": round(float(median_time), 3),
                "worst_case_time": round(float(worst_time), 3),
            })
            
        return sorted_predictions


class MonteCarloSimulator:
    """
    Vectorized NumPy Monte Carlo Race Simulator.
    Simulates 10,000 iterations of an F1 race in sub-second times.
    """
    def __init__(self, circuit_id):
        self.circuit_id = circuit_id
        self.circuit_meta = CIRCUITS[circuit_id]
        
    def simulate_race(self, starting_grid, tyre_strategies, num_sims=10000, current_lap=0, active_state=None, rain_intensity=0.0, constructor_pace_dynamic=None):
        """
        Runs 10,000 simulated race runs.
        
        starting_grid: list of driver codes in starting order (P1 to P20)
        tyre_strategies: dict mapping driver code to list of compounds (e.g. {"HAM": ["Medium", "Hard"]})
        current_lap: int (starts at 0, updates if live mid-race)
        active_state: dict containing live race gaps, tyre ages if running live
        rain_intensity: float (0.0: Dry, 0.5: Damp, 1.0: Wet)
        """
        num_drivers = len(starting_grid)
        total_laps = self.circuit_meta["laps"]
        
        # Override tyre strategies in wet conditions
        if rain_intensity == 0.5:
            tyre_strategies = {d: ["Intermediate", "Intermediate"] for d in starting_grid}
        elif rain_intensity == 1.0:
            tyre_strategies = {d: ["Wet", "Wet"] for d in starting_grid}
            
        deg_coeffs = self.circuit_meta["tyre_deg_coefficients"]
        
        # Safety Car & DNF multipliers
        sc_multiplier = 1.5 if rain_intensity == 0.5 else (2.5 if rain_intensity == 1.0 else 1.0)
        dnf_multiplier = 1.5 if rain_intensity == 0.5 else (2.5 if rain_intensity == 1.0 else 1.0)
        
        sc_base_prob = (self.circuit_meta["sc_probability"] * sc_multiplier) / total_laps # lap-by-lap SC probability
        dnf_base_prob = (self.circuit_meta["base_dnf_probability"] * dnf_multiplier) / total_laps # lap-by-lap DNF probability
        
        # Overtaking multiplier
        overtake_multiplier = 0.8 if rain_intensity == 0.5 else (0.6 if rain_intensity == 1.0 else 1.0)
        overtake_diff = self.circuit_meta["overtaking_index"] * overtake_multiplier
        
        # 1. State Matrix Initializations
        # Shape: (num_simulations, num_drivers)
        np.random.seed(42)
        
        # Base pace based on driver Elo rankings and constructor strength
        base_paces = np.zeros((num_sims, num_drivers))
        driver_consistencies = np.zeros(num_drivers) # standard deviation of lap times
        
        # Wet weather ELO adjustment configurations
        WET_EXPERTS = ["VER", "HAM", "ALO", "LEC"]
        
        for idx, d in enumerate(starting_grid):
            elo = GRID_2026[d]["base_elo"]
            
            # Temporary ELO adjustments for wet weather
            if rain_intensity == 0.5: # Damp
                if d in WET_EXPERTS:
                    elo += 50
                elif GRID_2026[d]["is_rookie"]:
                    elo -= 50
            elif rain_intensity == 1.0: # Wet
                if d in WET_EXPERTS:
                    elo += 100
                elif GRID_2026[d]["is_rookie"]:
                    elo -= 100
                    
            team_name = GRID_2026[d]["team"]
            if constructor_pace_dynamic and team_name in constructor_pace_dynamic:
                pace_offset = constructor_pace_dynamic[team_name].get("pace_offset", 0.0)
            else:
                pace_offset = CONSTRUCTORS_2026.get(team_name, {}).get("pace_offset", 0.0)
            
            # Base pace computation with weather delays (Damp: +8.0s, Wet: +16.0s)
            weather_pace_delay = 8.0 if rain_intensity == 0.5 else (16.0 if rain_intensity == 1.0 else 0.0)
            base_paces[:, idx] = 72.5 - (elo - 1500) * 0.003 + pace_offset + weather_pace_delay
            
            # Consistency multiplier: champions have smaller stdev, rookies have higher
            # Variance increases under rain, especially for rookies
            is_rookie = GRID_2026[d]["is_rookie"]
            base_consistency = 0.09 if not is_rookie else 0.18
            
            if rain_intensity == 0.5:
                consistency_scale = 1.5 if not is_rookie else 2.0
            elif rain_intensity == 1.0:
                consistency_scale = 2.0 if not is_rookie else 3.0
            else:
                consistency_scale = 1.0
                
            driver_consistencies[idx] = base_consistency * consistency_scale
            
        # Cumulative total race times (seconds)
        total_race_times = np.zeros((num_sims, num_drivers))
        
        # If live mid-race, initialize total race times with actual gaps
        if active_state is not None:
            # active_state: {"gaps": [0.0, 1.2, 5.4, ...], "tyre_ages": [10, 10, 15, ...]}
            gaps = np.array(active_state["gaps"])
            cumulative_gaps = np.cumsum(gaps)
            total_race_times += cumulative_gaps[np.newaxis, :]
            
        # Active Tyre compound compound index (0: Soft, 1: Medium, 2: Hard)
        # Active Tyre Age (laps on current tyre)
        # Active Pit Stop Counts
        tyre_compounds = np.zeros((num_sims, num_drivers), dtype=int) # index in strategy list
        tyre_ages = np.zeros((num_sims, num_drivers))
        pit_stops = np.zeros((num_sims, num_drivers))
        active_dnf = np.zeros((num_sims, num_drivers), dtype=bool) # True if driver has DNF'd
        
        if active_state is not None:
            tyre_ages += np.array(active_state["tyre_ages"])
            pit_stops += np.array(active_state["pit_stops"])
            # Initialize tyre compounds to the index matching the number of pit stops completed
            for idx, d in enumerate(starting_grid):
                pits = active_state["pit_stops"][idx]
                strat = tyre_strategies.get(d, ["Medium", "Hard"])
                tyre_compounds[:, idx] = np.clip(pits, 0, len(strat) - 1)
                
            if "dnfs" in active_state:
                for dnf_driver in active_state["dnfs"]:
                    if dnf_driver in starting_grid:
                        dnf_idx = starting_grid.index(dnf_driver)
                        active_dnf[:, dnf_idx] = True
            
        # 2. Main Simulation Loop (Lap-by-Lap)
        for lap in range(current_lap, total_laps):
            # A. Fuel Weight Drop (cars get ~0.04s faster per lap as fuel burns out)
            fuel_factor = (total_laps - lap) * -0.04
            
            # B. Tyre Wear degradation
            # Lap time penalty increases non-linearly with age: k * age^2
            tyre_penalty = np.zeros((num_sims, num_drivers))
            for idx, d in enumerate(starting_grid):
                strat = tyre_strategies.get(d, ["Medium", "Hard"])
                # Extract wear coefficients, defaulting if tyre type is not standard dry compounds
                strat_coeffs = np.array([deg_coeffs.get(c, 0.05 if c == "Intermediate" else 0.06) for c in strat])
                
                active_c_idx = np.clip(tyre_compounds[:, idx], 0, len(strat) - 1)
                k_wear = strat_coeffs[active_c_idx]
                tyre_penalty[:, idx] = k_wear * (tyre_ages[:, idx] ** 1.8) * 0.1
                    
            # C. Draw random lap noise (consistency)
            # Shape: (num_sims, num_drivers)
            epsilon = np.random.normal(0, driver_consistencies, size=(num_sims, num_drivers))
            
            # D. Lap Time Calculation
            # LapTime = BasePace + Fuel Drop + Tyre Wear + Driver Noise
            lap_times = base_paces + fuel_factor + tyre_penalty + epsilon
            
            # Add pit stop penalty if pit stop is triggered in this lap
            # Wet tyre strategy pit stop triggers: Inter at 24 laps, Wet at 20 laps
            for idx, d in enumerate(starting_grid):
                strat = tyre_strategies.get(d, ["Medium", "Hard"])
                strat_thresholds = np.array([16 if c == "Soft" else (26 if c == "Medium" else (38 if c == "Hard" else (24 if c == "Intermediate" else 20))) for c in strat])
                
                active_c_idx = np.clip(tyre_compounds[:, idx], 0, len(strat) - 1)
                thresholds = strat_thresholds[active_c_idx]
                
                pit_triggered = (tyre_ages[:, idx] >= thresholds) & (tyre_compounds[:, idx] < len(strat) - 1) & (~active_dnf[:, idx])
                
                if np.any(pit_triggered):
                    lap_times[pit_triggered, idx] += 22.0
                    tyre_ages[pit_triggered, idx] = 0
                    tyre_compounds[pit_triggered, idx] += 1
                    pit_stops[pit_triggered, idx] += 1
            
            # Apply lap time additions and increment tyre age
            total_race_times += lap_times
            tyre_ages += 1
            
            # E. DRS Telemetry & Traffic Overtaking Swap Limit (vectorized across simulations)
            sim_times_mask = np.where(active_dnf, 999999.0, total_race_times)
            sorted_positions = np.argsort(sim_times_mask, axis=1)
            
            for pos in range(1, num_drivers):
                lead_driver_idx = sorted_positions[:, pos - 1]
                chase_driver_idx = sorted_positions[:, pos]
                
                lead_times = total_race_times[np.arange(num_sims), lead_driver_idx]
                chase_times = total_race_times[np.arange(num_sims), chase_driver_idx]
                
                gaps = chase_times - lead_times
                in_drs = (gaps < 0.8) & (~active_dnf[np.arange(num_sims), chase_driver_idx]) & (~active_dnf[np.arange(num_sims), lead_driver_idx])
                overtake_roll = np.random.rand(num_sims) < (overtake_diff * 0.15)
                successful_overtake = in_drs & overtake_roll
                
                if np.any(successful_overtake):
                    idx_to_swap = np.where(successful_overtake)[0]
                    lead_idx = lead_driver_idx[idx_to_swap]
                    chase_idx = chase_driver_idx[idx_to_swap]
                    
                    temp = total_race_times[idx_to_swap, chase_idx]
                    total_race_times[idx_to_swap, chase_idx] = total_race_times[idx_to_swap, lead_idx] - 0.3
                    total_race_times[idx_to_swap, lead_idx] = temp + 0.3
                            
            # F. Safety Car Random Events
            # If SC is triggered (based on circuit base probability), compress the entire field
            # In each simulation run, SC has a small probability to trigger
            sc_triggered_sims = np.random.rand(num_sims) < sc_base_prob
            triggered_indices = np.where(sc_triggered_sims)[0]
            for sim_i in triggered_indices:
                sim_times = total_race_times[sim_i, :]
                sim_times_mask = np.where(active_dnf[sim_i, :], 999999.0, sim_times)
                sorted_idx = np.argsort(sim_times_mask)
                
                current_leader_time = sim_times_mask[sorted_idx[0]]
                valid_mask = ~active_dnf[sim_i, sorted_idx]
                valid_sorted_idx = sorted_idx[valid_mask]
                
                sc_gaps = np.zeros(len(valid_sorted_idx))
                if len(valid_sorted_idx) > 1:
                    sc_gaps[1:] = np.random.uniform(1.0, 1.5, size=len(valid_sorted_idx) - 1)
                cumulative_sc_gaps = np.cumsum(sc_gaps)
                total_race_times[sim_i, valid_sorted_idx] = current_leader_time + cumulative_sc_gaps
                        
            # G. Random Driver DNFs
            # Draws random mechanical or crash retirements
            dnf_drawn = np.random.rand(num_sims, num_drivers) < dnf_base_prob
            active_dnf = active_dnf | dnf_drawn
            # Penalize DNF'd drivers with massive total time so they rank last
            total_race_times = np.where(active_dnf, 999999.0, total_race_times)
            
        # 3. Post-Processing & Compilation of Results
        # Extract finishing rankings for all 10,000 simulations
        # Shape: (num_simulations, num_drivers)
        final_rankings = np.argsort(np.argsort(total_race_times, axis=1), axis=1) + 1
        
        # Build probability statistics per driver
        stats = {}
        for idx, d in enumerate(starting_grid):
            driver_ranks = final_rankings[:, idx]
            driver_dnf_count = np.sum(active_dnf[:, idx])
            
            p1_count = np.sum(driver_ranks == 1)
            podium_count = np.sum(driver_ranks <= 3)
            top10_count = np.sum(driver_ranks <= 10)
            
            stats[d] = {
                "driver_name": GRID_2026[d]["name"],
                "team": GRID_2026[d]["team"],
                "win_probability": round((p1_count / num_sims) * 100, 2),
                "podium_probability": round((podium_count / num_sims) * 100, 2),
                "top10_probability": round((top10_count / num_sims) * 100, 2),
                "dnf_probability": round((driver_dnf_count / num_sims) * 100, 2),
                "rank_distribution": [int(np.sum(driver_ranks == pos)) for pos in range(1, num_drivers + 1)]
            }
            
        return stats
