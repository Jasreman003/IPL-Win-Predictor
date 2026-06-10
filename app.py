import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Probability Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}

.main { background: #0a0e1a; }

h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }

.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%); }

/* Hero title */
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f97316, #fbbf24, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 0;
}

.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

.metric-card h4 { color: #94a3b8; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; margin: 0; }
.metric-card .val { font-family: 'Rajdhani', sans-serif; font-size: 2.2rem; font-weight: 700; color: #f8fafc; margin: 0; }

/* Probability gauge */
.gauge-container {
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

.gauge-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 1rem;
}

.prob-number {
    font-family: 'Rajdhani', sans-serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
}

/* Divider */
.ipl-divider {
    border: none;
    border-top: 1px solid #1e3a5f;
    margin: 1.5rem 0;
}

/* Stat badges */
.stat-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 2px;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, #0f2027, #1e3a5f40);
    border-left: 3px solid #f97316;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.9rem;
}

/* Team pill */
.team-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
            
.input-label {
    color: #93c5fd;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 5px;
} 

.streamlit-expanderHeader {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* For newer Streamlit versions */
details summary {
    color: #ffffff !important;
    font-weight: 600 !important;
}
                                   
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model" / "ipl_model.pkl"
ENCODER_PATH = Path(__file__).parent / "model" / "encoders.pkl"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists() and ENCODER_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoders = pickle.load(f)
        return model, encoders
    return None, None

model, encoders = load_model()

# ── Teams & Venues ────────────────────────────────────────────────────────────
TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants",
    "Gujarat Titans"
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

TEAM_COLORS = {
    "Mumbai Indians": "#004BA0",
    "Chennai Super Kings": "#F9CD05",
    "Royal Challengers Bangalore": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Delhi Capitals": "#0078BC",
    "Sunrisers Hyderabad": "#F7A721",
    "Rajasthan Royals": "#EA1A7F",
    "Punjab Kings": "#AA4545",
    "Lucknow Super Giants": "#A0EEE2",
    "Gujarat Titans": "#1C1C60",
}

# ── Prediction logic ──────────────────────────────────────────────────────────
def predict_win_probability(batting_team, bowling_team, venue, target,
                             current_score, current_over, current_wickets):
    """Compute win probability using model or heuristic fallback."""
    balls_remaining = (20 - current_over) * 6
    runs_needed = target - current_score
    wickets_remaining = 10 - current_wickets
    current_rr = current_score / max(current_over, 0.1)
    required_rr = (runs_needed / max(balls_remaining, 1)) * 6

    # ── If model is loaded ──
    if model is not None and encoders is not None:
        try:
            bat_enc = encoders["team"].transform([batting_team])[0]
            bowl_enc = encoders["team"].transform([bowling_team])[0]
            venue_enc = encoders["venue"].transform([venue])[0]
            features = np.array([[bat_enc, bowl_enc, venue_enc, target,
                                   current_score, current_over,
                                   runs_needed, current_rr,
                                   required_rr, balls_remaining]])
            prob = model.predict_proba(features)[0][1]
            return float(prob), "model"
        except Exception:
            pass  # Fall through to heuristic

    # ── Heuristic fallback ──
    if runs_needed <= 0:
        return 1.0, "heuristic"
    if balls_remaining <= 0:
        return 0.0, "heuristic"
    if wickets_remaining <= 0:
        return 0.0, "heuristic"

    rrr_ratio = required_rr / max(current_rr, 1.0)
    wicket_factor = wickets_remaining / 10.0
    progress = current_over / 20.0

    raw = (1 / (1 + np.exp(2 * (rrr_ratio - 1)))) * (0.5 + 0.5 * wicket_factor)
    raw = raw * (1 - 0.15 * (1 - progress))
    raw = float(np.clip(raw, 0.04, 0.96))
    return raw, "heuristic"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏏 About")
    st.markdown("""
    <div style='color:#94a3b8; font-size:0.85rem; line-height:1.7'>
    This predictor uses a <strong style='color:#f97316'>Logistic Regression</strong>
    model trained on <strong style='color:#fbbf24'>10 seasons</strong> of IPL
    ball-by-ball data.<br><br>
    <strong>Features used:</strong>
    </div>
    """, unsafe_allow_html=True)
    features_list = ["Run rate & required rate", "Wickets in hand", "Balls remaining",
                     "Venue (home advantage)", "Team encoding", "Target score"]
    for f in features_list:
        st.markdown(f'<span class="stat-badge">✦ {f}</span>', unsafe_allow_html=True)

    st.markdown("<hr class='ipl-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#94a3b8; font-size:0.82rem'>
    📊 <strong style='color:#1e3a5f'>Model accuracy:</strong> <span style='color:#4ade80'>79%</span><br>
    📉 <strong style='color:#1e3a5f'>Naive baseline:</strong> <span style='color:#f87171'>61%</span><br>
    🧪 <strong style='color:#1e3a5f'>Test set:</strong> 2022 IPL season<br>
    🗃️ <strong style='color:#1e3a5f'>Training data:</strong> Kaggle ball-by-ball
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='ipl-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#64748b; font-size:0.78rem'>
    Run <code>python train_model.py</code> first to generate the trained model.
    Without the model file, a statistical heuristic is used.
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🏏 IPL WIN PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real-time In-Match Win Probability Engine</div>', unsafe_allow_html=True)

model_status = "✅ ML Model Loaded" if model is not None else "⚡ Heuristic Mode (run train_model.py)"
status_color = "#4ade80" if model is not None else "#fbbf24"
st.markdown(f'<div style="text-align:center; margin-bottom:1.5rem"><span style="background:#1e293b; border:1px solid #334155; border-radius:20px; padding:4px 16px; color:{status_color}; font-size:0.82rem; letter-spacing:1px">{model_status}</span></div>', unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("""
            <h3 style='color:white; font-weight:700;'>
            ⚙️ Match State Input
            </h3>
            """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-label">🟦 Batting Team</div>', unsafe_allow_html=True)
    batting_team = st.selectbox("Batting Team", TEAMS, index=0, label_visibility="collapsed")

    st.markdown('<div class="input-label">🏟️ Venue</div>', unsafe_allow_html=True)
    venue = st.selectbox("Venue", VENUES, index=0, label_visibility="collapsed")

    st.markdown('<div class="input-label">🎯 Target Score</div>', unsafe_allow_html=True)
    target = st.number_input("Target", min_value=50, max_value=300, value=175, step=1, label_visibility="collapsed")

    st.markdown('<div class="input-label">📊 Current Score</div>', unsafe_allow_html=True)
    current_score = st.number_input("Current Score", min_value=0, max_value=300, value=80, step=1, label_visibility="collapsed")

with col2:
    st.markdown('<div class="input-label">🟥 Bowling Team</div>', unsafe_allow_html=True)
    bowling_options = [t for t in TEAMS if t != batting_team]
    bowling_team = st.selectbox("Bowling Team", bowling_options, index=0, label_visibility="collapsed")

    st.markdown('<div class="input-label">🕐 Current Over</div>', unsafe_allow_html=True)
    current_over = st.slider("Current Over", min_value=0.1, max_value=19.5, value=10.0, step=0.1, label_visibility="collapsed")

    st.markdown('<div class="input-label">❌ Wickets Lost</div>', unsafe_allow_html=True)
    current_wickets = st.slider("Wickets Lost", min_value=0, max_value=9, value=3, step=1, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  PREDICT WIN PROBABILITY", width="stretch", type="primary")

# ── Live stats preview ────────────────────────────────────────────────────────
runs_needed = max(target - current_score, 0)
balls_left = int((20 - current_over) * 6)
crr = current_score / max(current_over, 0.1)
rrr = (runs_needed / max(balls_left, 1)) * 6
wickets_rem = 10 - current_wickets

st.markdown("<hr class='ipl-divider'>", unsafe_allow_html=True)
st.markdown("""
            <h3 style='color:white; font-weight:700;'>
            📈 Live Match Stats
            </h3>
            """, unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
stats = [
    (m1, "Runs Needed", str(runs_needed), "#f97316"),
    (m2, "Balls Left", str(balls_left), "#60a5fa"),
    (m3, "Wickets Left", str(wickets_rem), "#4ade80"),
    (m4, "Curr RR", f"{crr:.2f}", "#a78bfa"),
    (m5, "Req RR", f"{rrr:.2f}", "#f472b6"),
]
for col, label, val, color in stats:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <p class="val" style="color:{color}">{val}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Result ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='ipl-divider'>", unsafe_allow_html=True)

if predict_btn or True:  # Always show on load with defaults
    prob, method = predict_win_probability(
        batting_team, bowling_team, venue, target,
        current_score, current_over, current_wickets
    )
    bat_prob = prob
    bowl_prob = 1 - prob

    col_bat, col_mid, col_bowl = st.columns([2, 3, 2])

    bat_color = TEAM_COLORS.get(batting_team, "#f97316")
    bowl_color = TEAM_COLORS.get(bowling_team, "#60a5fa")

    with col_bat:
        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-title">🏏 BATTING</div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; color:#94a3b8; margin-bottom:0.5rem">{batting_team}</div>
            <div class="prob-number" style="color:{bat_color}">{bat_prob*100:.1f}%</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:0.5rem">WIN PROBABILITY</div>
        </div>
        """, unsafe_allow_html=True)

    with col_mid:
        # Bar gauge
        bar_pct = int(bat_prob * 100)
        st.markdown(f"""
        <div class="gauge-container" style="padding: 1.5rem 2rem">
            <div class="gauge-title">⚡ HEAD-TO-HEAD</div>
            <div style="margin: 1.2rem 0">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px">
                    <span style="color:{bat_color}; font-size:0.8rem; font-weight:700">{batting_team[:3].upper()}</span>
                    <span style="color:{bowl_color}; font-size:0.8rem; font-weight:700">{bowling_team[:3].upper()}</span>
                </div>
                <div style="background:#1e293b; border-radius:999px; height:14px; overflow:hidden; border:1px solid #334155">
                    <div style="width:{bar_pct}%; background:linear-gradient(90deg, {bat_color}, {bat_color}cc); height:100%; border-radius:999px; transition:all 0.5s ease"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:6px">
                    <span style="color:#94a3b8; font-size:0.75rem">{bat_prob*100:.1f}%</span>
                    <span style="color:#94a3b8; font-size:0.75rem">{bowl_prob*100:.1f}%</span>
                </div>
            </div>
            <div style="margin-top:1rem">
        """, unsafe_allow_html=True)

        verdict = batting_team if bat_prob >= 0.5 else bowling_team
        conf = max(bat_prob, bowl_prob)
        conf_label = "HIGH" if conf > 0.70 else "MODERATE" if conf > 0.55 else "CLOSE"
        conf_color = "#4ade80" if conf > 0.70 else "#fbbf24" if conf > 0.55 else "#f87171"

        st.markdown(f"""
                <div style="text-align:center; margin-top:0.5rem">
                    <div style="color:#94a3b8; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase">Predicted Winner</div>
                    <div style="font-family:'Rajdhani',sans-serif; font-size:1.5rem; font-weight:700; color:#f8fafc; margin:4px 0">{verdict}</div>
                    <span style="background:{conf_color}22; color:{conf_color}; border:1px solid {conf_color}55; border-radius:20px; padding:2px 12px; font-size:0.75rem; letter-spacing:2px">{conf_label} CONFIDENCE</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_bowl:
        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-title">🎳 BOWLING</div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1rem; color:#94a3b8; margin-bottom:0.5rem">{bowling_team}</div>
            <div class="prob-number" style="color:{bowl_color}">{bowl_prob*100:.1f}%</div>
            <div style="color:#64748b; font-size:0.8rem; margin-top:0.5rem">WIN PROBABILITY</div>
        </div>
        """, unsafe_allow_html=True)

    method_label = "🤖 Logistic Regression Model" if method == "model" else "📐 Statistical Heuristic"
    st.markdown(f'<div style="text-align:center; margin-top:1rem"><span style="color:#475569; font-size:0.78rem">{method_label} • Predicted in &lt;3s</span></div>', unsafe_allow_html=True)

# ── Scenario explorer ─────────────────────────────────────────────────────────
st.markdown("<hr class='ipl-divider'>", unsafe_allow_html=True)
with st.expander("🔬 Scenario Explorer — How does probability shift over overs?"):
    import plotly.graph_objects as go

    scenario_overs = np.arange(0.5, 20.0, 0.5)
    scenario_probs = []
    for ov in scenario_overs:
        # Simulate linear scoring
        sim_score = int((current_score / max(current_over, 0.1)) * ov)
        sim_score = min(sim_score, target - 1)
        sim_wkts = min(int(current_wickets * ov / max(current_over, 0.1)), 9)
        p, _ = predict_win_probability(batting_team, bowling_team, venue, target,
                                        sim_score, ov, sim_wkts)
        scenario_probs.append(p * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(scenario_overs), y=scenario_probs,
        mode='lines+markers',
        line=dict(color='#f97316', width=2.5),
        marker=dict(size=5, color='#fbbf24'),
        fill='tozeroy',
        fillcolor='rgba(249,115,22,0.08)',
        name=batting_team,
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#475569",
                  annotation_text="50% line", annotation_position="right")
    fig.add_vline(x=current_over, line_dash="dot", line_color="#60a5fa",
                  annotation_text=f"Now ({current_over})", annotation_position="top right")

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.8)',
        font=dict(color='#94a3b8', family='Exo 2'),
        xaxis=dict(title='Over', gridcolor='#1e293b', color='#64748b'),
        yaxis=dict(title='Win Probability (%)', range=[0, 100], gridcolor='#1e293b', color='#64748b'),
        margin=dict(l=0, r=0, t=20, b=0),
        height=320,
    )
    st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#94a3b8'>
    Developed by <b>Jasreman Kaur</b> 
    </div>
    """,
    unsafe_allow_html=True
)