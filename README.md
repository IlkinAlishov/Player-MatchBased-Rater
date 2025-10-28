# ⚽ Player Match-Based Rater🌟

## Project Overview

This repository hosts an end-to-end Machine Learning pipeline designed to predict football player match ratings. The project serves as a robust demonstration of core MLOps principles, showcasing:

- **Data Ingestion:** Reading and merging disparate data sources (CSV and SQLite).
- **Feature Engineering:** Comprehensive feature creation, including critical per-90 metrics.
- **Model Training:** Training, persistence (.pkl), and systematic evaluation.
- **Automation:** Full pipeline execution via a single shell script (`run_all.sh`).

The model aims to predict player performance scores based on detailed in-game statistics.

## ⚙️ Repository Structure

The project adheres to a standard, scalable directory structure:

```

Player-MatchBased-Rater/
├── data/                        # Raw data files (tracked by Git LFS)
│   └── raw/                     # Raw data (CSV, SQLite, etc.)
├── models/                      # Saved trained model artifacts (Ignored by Git)
├── reports/                     # Performance metrics, plots, and reports (Ignored by Git)
├── src/
│   ├── eval/                    # Evaluation logic (e.g., full_evaluate.py)
│   ├── features/                # Cleaning and feature engineering (prepare_data.py)
│   ├── ingest/                  # Data loading and merging (ingest_data.py)
│   └── model/                   # Model training and saving (train_model.py)
├── .gitignore                   # Files excluded from Git tracking (models, reports, large processed data)
├── .gitattributes               # Git LFS configuration for raw data
├── requirements.txt             # Project dependencies
└── run_all.sh                   # The main execution script

````

## 🚀 Getting Started

Follow these steps to set up the project locally and run the full pipeline.

### Prerequisites

- **Git & Git LFS:** Ensure both Git and Git LFS (Large File Storage) are installed on your machine.

    ```bash
    git lfs install
    ```

- **Python:** Python 3.9 or higher.

### Setup Instructions

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/IlkinAlishov/Player-MatchBased-Rater.git
    cd Player-MatchBased-Rater
    ```

2. **Create and Activate Environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Execution

The entire pipeline, from raw data processing to final model evaluation, is executed with a single command.

### Run the Full Pipeline

```bash
./run_all.sh
````

### Pipeline Flow

| Step             | Script                         | Purpose                                                                                |
| ---------------- | ------------------------------ | -------------------------------------------------------------------------------------- |
| **Ingest**       | `src/ingest/ingest_data.py`    | Combines data from `.csv` and `.sqlite` files.                                         |
| **Prepare Data** | `src/features/prepare_data.py` | Applies cleaning, handles missing data, and creates features like `goals_per90`.       |
| **Train**        | `src/model/train_model.py`     | Trains the regressor model and saves the trained artifact to `models/`.                |
| **Evaluate**     | `src/eval/full_evaluate.py`    | Calculates granular metrics (e.g., MAE by position and rating band) and saves reports. |

## 📊 Results and Output

Upon successful execution, the following artifacts are generated locally:

* **Key Metrics** (Example from `reports/metrics.txt`):

  | Metric                            | Performance |
  | --------------------------------- | ----------- |
  | **MAE** (Mean Absolute Error)     | 0.2876      |
  | **RMSE** (Root Mean Square Error) | 0.3849      |

* **MAE by Rating Band**:

  | Rating Band | MAE     | Samples (n) |
  | ----------- | ------- | ----------- |
  | ≈6          | 0.2303  | 964         |
  | ≈7          | 0.2512  | 993         |
  | ≥8          | 0.5015  | 403         |

* **MAE by Position**:

  | Position | MAE     | Samples (n) |
  | -------- | ------- | ----------- |
  | DF       | 0.3384  | 1188        |
  | FW       | 0.2969  | 527         |
  | GK       | 0.3379  | 320         |
  | MF       | 0.2985  | 1423        |
  | Sub      | 0.1684  | 813         |

* **MAE by Competition (Top 10)**:

  | Competition                | MAE     | Samples (n) |
  | -------------------------- | ------- | ----------- |
  | World Cup 2018             | 0.2848  | 330         |
  | Bundesliga 2017-18         | 0.2854  | 1626        |
  | Premier League 2017-18     | 0.2895  | 2084        |
  | Euro 2016                  | 0.2903  | 231         |

* **Output Files**:
    - **Model:** `models/rating_model.pkl`
    - **Full Evaluation Report:** `reports/full_eval.txt` (Contains detailed breakdown of MAE by position, rating band, and competition).
    - **Visualizations:** `reports/residual_hist.png` (Visualization of model error distribution).


* **Output Files**:

  * **Model:** `models/rating_model.pkl`
  * **Full Evaluation Report:** `reports/full_eval.txt` (Contains detailed breakdown of MAE by position and rating band).
  * **Visualizations:** `reports/residual_hist.png` (Visualization of model error distribution).


## ✍️ Contribution

If you have ideas for new features, better prompts, or cleaner code, feel free to submit a **Pull Request** or open an **Issue**!

---



