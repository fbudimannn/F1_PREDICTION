"""
Script to update fastf1_tutorial.ipynb with new Season ELO, Dynamic Constructor Pace,
and ML Training pipeline content.

Usage:
    python update_notebook.py
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOTEBOOK_PATH = "fastf1_tutorial.ipynb"

# New cells to insert into the notebook
NEW_CELLS = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 🔄 Season ELO Carryover\n",
            "\n",
            "Instead of static `base_elo` values, the system now **accumulates driver ELO ratings** across all completed 2026 GPs.\n",
            "\n",
            "For each completed GP:\n",
            "- **Qualifying results** update ELO with **60% weight**\n",
            "- **Race results** update ELO with **40% weight**\n",
            "- **K-factor = 24** (lower than standard 32 for season-long stability)\n",
            "\n",
            "```\n",
            "Season Start:     Franco ELO = 1580 (base)\n",
            "After GP AUS: Franco ELO = 1580 + update AUS = 1597\n",
            "After GP CHN: Franco ELO = 1597 + update CHN = 1594\n",
            "After GP JPN: Franco ELO = 1594 + update JPN = 1611\n",
            "```\n",
            "\n",
            "The Season ELO is **cumulative** — each GP builds on the previous posterior."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Demonstrate Season ELO Carryover\n",
            "from src.season_elo import compute_season_elo, compute_dynamic_constructor_pace\n",
            "from src.utils import get_initial_driver_priors\n",
            "\n",
            "# Compute cumulative ELO up to a specific circuit\n",
            "season_elo = compute_season_elo('monaco')  # All GPs before Monaco\n",
            "base_elo = get_initial_driver_priors()\n",
            "\n",
            "# Show ELO changes from base\n",
            "print('Season ELO Carryover (up to Monaco GP):')\n",
            "print('-' * 60)\n",
            "print(f'{\"Driver\":<6} | {\"Base ELO\":<10} | {\"Season ELO\":<12} | {\"Delta\":<8}')\n",
            "print('-' * 60)\n",
            "for d in sorted(season_elo, key=lambda x: season_elo[x], reverse=True)[:10]:\n",
            "    delta = season_elo[d] - base_elo[d]\n",
            "    sign = '+' if delta >= 0 else ''\n",
            "    print(f'{d:<6} | {base_elo[d]:<10} | {season_elo[d]:<12} | {sign}{delta}')"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 🔧 Dynamic Constructor Pace Inference\n",
            "\n",
            "Constructor pace offsets are **dynamically inferred** from real qualifying data:\n",
            "- Uses **rolling 3-GP window** of gap-to-pole data\n",
            "- Detects **upgrade/downgrade trends** automatically\n",
            "- Replaces static hardcoded `CONSTRUCTORS_2026` pace offsets"
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Demonstrate Dynamic Constructor Pace\n",
            "dynamic_pace = compute_dynamic_constructor_pace('monaco')\n",
            "\n",
            "print('Dynamic Constructor Pace (up to Monaco GP):')\n",
            "print('-' * 60)\n",
            "print(f'{\"Team\":<18} | {\"Dynamic Offset\":<16} | {\"Trend\":<10}')\n",
            "print('-' * 60)\n",
            "for team in sorted(dynamic_pace, key=lambda x: dynamic_pace[x].get(\"pace_offset\", 0)):\n",
            "    data = dynamic_pace[team]\n",
            "    offset = data.get('pace_offset', 0)\n",
            "    trend = data.get('trend', 'unknown')\n",
            "    sign = '+' if offset >= 0 else ''\n",
            "    print(f'{team:<18} | {sign}{offset:<15.3f} | {trend}')"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 🧠 ML Model Training & Loading\n",
            "\n",
            "The LightGBM models are now **properly trained** using historical data (2022-2025) with:\n",
            "- **Time-Series Cross-Validation** (no data leakage)\n",
            "- **Optuna hyperparameter tuning** (30 trials)\n",
            "- **Anti-overfitting guards** (L1/L2 regularization, early stopping)\n",
            "\n",
            "Models are saved as `.joblib` files in the `models/` directory."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Load pre-trained models and display metrics\n",
            "import json\n",
            "import os\n",
            "\n",
            "metrics_path = os.path.join('models', 'training_metrics.json')\n",
            "params_path = os.path.join('models', 'hyperparameters.json')\n",
            "\n",
            "if os.path.exists(metrics_path):\n",
            "    with open(metrics_path) as f:\n",
            "        metrics = json.load(f)\n",
            "    print('Training Metrics:')\n",
            "    print(f'  Train years: {metrics[\"train_years\"]}')\n",
            "    print(f'  Test year: {metrics[\"test_year\"]}')\n",
            "    print(f'  Train samples: {metrics[\"train_samples\"]}')\n",
            "    print(f'  Test samples: {metrics[\"test_samples\"]}')\n",
            "    print(f'  Test NDCG@5: {metrics[\"test_ndcg5\"]:.4f}')\n",
            "    print(f'  Overfit status: {metrics[\"overfit_status\"]}')\n",
            "else:\n",
            "    print('No training metrics found. Run: python train_model.py')\n",
            "\n",
            "if os.path.exists(params_path):\n",
            "    with open(params_path) as f:\n",
            "        params = json.load(f)\n",
            "    print(f'\\nBest Hyperparameters:')\n",
            "    for k, v in params.items():\n",
            "        print(f'  {k}: {v}')"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Load and test the trained qualifying model\n",
            "from src.models import QualifyingModel\n",
            "\n",
            "model = QualifyingModel()\n",
            "model.load_trained_models()\n",
            "\n",
            "print(f'Model loaded: {\"trained\" if model._using_trained_models else \"mock fallback\"}')\n",
            "print(f'Ranker: {type(model.ranker).__name__}')\n",
            "print(f'Quantile models: {len(model.quantile_models)} loaded')"
        ],
        "outputs": [],
        "execution_count": None
    }
]


def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found")
        sys.exit(1)

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find insertion point: after the last existing cell
    cells = nb.get("cells", [])

    # Check if we already added these cells (idempotency check)
    existing_sources = [
        "".join(c.get("source", [])) for c in cells
    ]
    if any("Season ELO Carryover" in s and "compute_season_elo" in s for s in existing_sources):
        print("Notebook already contains Season ELO cells. Skipping update.")
        return

    # Insert new cells at the end
    for new_cell in NEW_CELLS:
        cells.append(new_cell)

    nb["cells"] = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Successfully updated {NOTEBOOK_PATH}")
    print(f"  Added {len(NEW_CELLS)} new cells (markdown + code)")
    print(f"  Total cells now: {len(cells)}")


if __name__ == "__main__":
    main()
