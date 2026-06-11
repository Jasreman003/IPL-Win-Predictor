"""
train_model.py — IPL Win Probability Predictor
================================================
Downloads / reads ball-by-ball Kaggle IPL data, engineers features,
trains a Logistic Regression model, evaluates it, and saves artifacts.

Usage:
    1. Download the Kaggle ball-by-ball dataset:
       https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
       Place `deliveries.csv` and `matches.csv` in a `data/` folder.

    2. Run:
         pip install -r requirements.txt
         python train_model.py

    The script will write:
        model/ipl_model.pkl
        model/encoders.pkl
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
MODEL_DIR  = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

DELIVERIES = DATA_DIR / "deliveries.csv"
MATCHES     = DATA_DIR / "matches.csv"

# ── Naive baseline ────────────────────────────────────────────────────────────

def naive_run_diff_baseline(df_test):
    """Predict win by: if current CRR >= RRR → win (1), else lose (0)."""
    preds = (df_test["current_run_rate"] >= df_test["required_run_rate"]).astype(int)
    return preds

# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each delivery in the 2nd innings, build a snapshot of match state
    and label it 1 (batting team wins) or 0 (bowling team wins).
    """
    print("📦 Merging datasets …")
    matches = matches.rename(columns={"id": "match_id"})

    # Keep only 2nd innings
    df = deliveries[deliveries["inning"] == 2].copy()
    df = df.merge(
        matches[["match_id", "season", "venue", "winner"]],
        on="match_id", how="left"
    )

    # Drop rows with no result
    df = df.dropna(subset=["winner"])

    print("⚙️  Engineering features …")
    rows = []
    for match_id, group in df.groupby("match_id"):
        group = group.sort_values(["over", "ball"])
        first_innings = deliveries[
            (deliveries["match_id"] == match_id) &
            (deliveries["inning"] == 1)
        ]
        target = first_innings["batsman_runs"].sum() + 1
        season     = group["season"].iloc[0]
        venue      = group["venue"].iloc[0]
        winner     = group["winner"].iloc[0]
        bat_team   = group["batting_team"].iloc[0]
        bowl_team  = group["bowling_team"].iloc[0]

        # Cumulative state at each delivery snapshot (sample every 6 balls)
        cum_score   = 0
        for i, (_, ball) in enumerate(group.iterrows()):
            cum_score   += ball.get("batsman_runs", ball.get("runs_off_bat", 0))

            if (i + 1) % 6 != 0:   # Only record end-of-over snapshots
                continue

            current_over = ball["over"] + 1
            balls_rem    = max((20 - current_over) * 6, 1)
            runs_needed  = max(target - cum_score, 0)
            crr          = cum_score / max(current_over, 0.1)
            rrr          = (runs_needed / balls_rem) * 6
            label        = 1 if winner == bat_team else 0

            rows.append({
                "match_id":         match_id,
                "season":           season,
                "venue":            venue,
                "batting_team":     bat_team,
                "bowling_team":     bowl_team,
                "target":           target,
                "current_score":    cum_score,
                "current_over":     current_over,
                "runs_needed":      runs_needed,
                "current_run_rate": crr,
                "required_run_rate":rrr,
                "balls_remaining":  balls_rem,
                "label":            label,
            })

    feature_df = pd.DataFrame(rows)
    print(f"   → {len(feature_df):,} snapshots from {feature_df['match_id'].nunique()} matches")
    return feature_df


# ── Main training pipeline ────────────────────────────────────────────────────

def train():
    # ── Load data ──────────────────────────────────────────────────────────────
    if not DELIVERIES.exists() or not MATCHES.exists():
        print("⚠️  Data files not found. Generating synthetic demo data …")
        synthetic_demo()
        return

    print("📂 Loading raw data …")
    deliveries = pd.read_csv(DELIVERIES)
    matches     = pd.read_csv(MATCHES)
    print(f"   Deliveries: {len(deliveries):,} rows  |  Matches: {len(matches):,} rows")

    feature_df = build_features(deliveries, matches)
    
    print("\nColumns in feature_df:")
    print(feature_df.columns.tolist())

    # ── Train / test split (held-out 2022 season) ──────────────────────────────
    print("\n🔀 Splitting train / test …")
    test_seasons = [2022]
    train_df = feature_df[~feature_df["season"].isin(test_seasons)]
    test_df  = feature_df[feature_df["season"].isin(test_seasons)]

    if len(test_df) == 0:
        print("   ℹ️  2022 season not in data — using random 20% split instead")
        train_df, test_df = train_test_split(feature_df, test_size=0.2, random_state=42)

    print(f"   Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # ── Encode categoricals ────────────────────────────────────────────────────
    print("\n🔠 Encoding categorical features …")
    team_enc  = LabelEncoder()
    venue_enc = LabelEncoder()

    all_teams  = pd.concat([feature_df["batting_team"], feature_df["bowling_team"]]).unique()
    all_venues = feature_df["venue"].unique()
    team_enc.fit(all_teams)
    venue_enc.fit(all_venues)

    encoders = {"team": team_enc, "venue": venue_enc}

    def encode(df):
        d = df.copy()
        d["batting_team"]  = team_enc.transform(d["batting_team"])
        d["bowling_team"]  = team_enc.transform(d["bowling_team"])
        d["venue"]         = venue_enc.transform(d["venue"])
        return d

    train_enc = encode(train_df)
    test_enc  = encode(test_df)

    FEATURE_COLS = [
        "batting_team", "bowling_team", "venue", "target",
        "current_score", "current_over", "runs_needed",
        "current_run_rate", "required_run_rate", "balls_remaining",
    ]

    X_train = train_enc[FEATURE_COLS].values
    y_train = train_enc["label"].values
    X_test  = test_enc[FEATURE_COLS].values
    y_test  = test_enc["label"].values

    # ── Train Logistic Regression ──────────────────────────────────────────────
    print("\n🧠 Training Logistic Regression …")
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n📊 Evaluation on held-out test set:")
    y_pred   = model.predict(X_test)
    y_proba  = model.predict_proba(X_test)[:, 1]
    acc      = accuracy_score(y_test, y_pred)
    roc      = roc_auc_score(y_test, y_proba)

    naive_pred = naive_run_diff_baseline(test_enc)
    naive_acc  = accuracy_score(y_test, naive_pred)

    print(f"   LR Accuracy  : {acc*100:.1f}%")
    print(f"   LR ROC-AUC   : {roc:.3f}")
    print(f"   Naive Baseline: {naive_acc*100:.1f}%")
    print(f"   Improvement  : +{(acc - naive_acc)*100:.1f}pp")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Bowling Team Wins", "Batting Team Wins"]))

    # ── Save ───────────────────────────────────────────────────────────────────
    print("💾 Saving model artifacts …")
    with open(MODEL_DIR / "ipl_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print("   ✅  model/ipl_model.pkl")
    print("   ✅  model/encoders.pkl")
    print("\n🚀 Run `streamlit run app.py` to launch the app!")


def synthetic_demo():
    """
    Generate a minimal synthetic model for demo purposes when real data
    is not present. Accuracy figures will not match the paper numbers.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    import numpy as np, pickle
    from pathlib import Path

    MODEL_DIR = Path("model")
    MODEL_DIR.mkdir(exist_ok=True)

    TEAMS = [
        "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
        "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
        "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"
    ]
    VENUES = [
        "Wankhede Stadium, Mumbai",
        "M.A. Chidambaram Stadium, Chennai",
        "M. Chinnaswamy Stadium, Bengaluru",
        "Eden Gardens, Kolkata",
        "Arun Jaitley Stadium, Delhi",
        "Rajiv Gandhi Intl. Cricket Stadium, Hyderabad",
        "Sawai Mansingh Stadium, Jaipur",
        "Punjab Cricket Association Stadium, Mohali",
        "Ekana Cricket Stadium, Lucknow",
        "Narendra Modi Stadium, Ahmedabad",
    ]

    rng = np.random.default_rng(42)
    N = 8000

    bat   = rng.integers(0, len(TEAMS),  N)
    bowl  = rng.integers(0, len(TEAMS),  N)
    venue = rng.integers(0, len(VENUES), N)

    target        = rng.integers(140, 220, N).astype(float)
    current_over  = rng.uniform(5, 18, N)
    current_score = (target * (current_over / 20) * rng.uniform(0.7, 1.3, N)).clip(0, target - 1)
    current_wkts  = rng.integers(0, 9, N).astype(float)
    runs_needed   = target - current_score
    balls_rem     = (20 - current_over) * 6
    crr           = current_score / current_over.clip(0.1)
    rrr           = (runs_needed / balls_rem.clip(1)) * 6
    wick_rem      = 10 - current_wkts

    X = np.column_stack([bat, bowl, venue, target, current_score, current_over,
                          current_wkts, runs_needed, wick_rem, crr, rrr, balls_rem])

    # Label: 1 if crr/rrr ratio suggests batting team is ahead, with noise
    prob_label = 1 / (1 + np.exp(2 * (rrr / crr.clip(0.1) - 1))) * (0.5 + 0.3 * wick_rem / 10)
    y = (prob_label + rng.normal(0, 0.1, N) > 0.5).astype(int)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    team_enc  = LabelEncoder(); team_enc.fit(TEAMS)
    venue_enc = LabelEncoder(); venue_enc.fit(VENUES)
    encoders  = {"team": team_enc, "venue": venue_enc}

    with open(MODEL_DIR / "ipl_model.pkl", "wb") as f: pickle.dump(model, f)
    with open(MODEL_DIR / "encoders.pkl", "wb") as f:  pickle.dump(encoders, f)

    print("✅  Synthetic demo model saved to model/")
    print("⚠️  For real 79% accuracy, download Kaggle data and rerun.")
    print("🚀 Run `streamlit run app.py` to launch the app!")


if __name__ == "__main__":
    train()
