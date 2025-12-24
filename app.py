import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# CONFIG (chốt theo chat)
# =========================
N_POINTS = 15

# Normalization constants (K)
K_HRV = 80.0     # %ΔHRV / 80
K_RR  = 25.0     # %ΔRR  / 25
K_SPO2 = 5.0     # ΔSpO2 / 5 (points)

# STEP emergency thresholds (between 2 points)
HRV_STEP_RED_PCT = -40.0   # %ΔHRV <= -40% => red
RR_STEP_RED_PCT  = +30.0   # %ΔRR  >= +30% => red
SPO2_STEP_RED_PT = -4.0    # ΔSpO2 <= -4 points => red

# Fusion-based step thresholds
DN_STEP_WARN = 0.25
DN_STEP_RED  = 0.55

# Rolling (trend) thresholds
DN_ROLL_WARN = 0.18
DN_ROLL_RED  = 0.35

# =========================
# HELPERS
# =========================
def parse_series(text: str, n: int) -> np.ndarray:
    """
    Accepts space/comma/newline separated numbers.
    Returns length-n float array (pads or trims).
    """
    if text is None:
        return np.zeros(n, dtype=float)
    parts = (
        text.replace(",", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .split()
    )
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except:
            pass
    if len(vals) == 0:
        vals = [0.0] * n
    if len(vals) < n:
        vals = vals + [vals[-1]] * (n - len(vals))
    if len(vals) > n:
        vals = vals[:n]
    return np.array(vals, dtype=float)

def pct_change(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        prev = x[i-1]
        if prev == 0:
            d[i] = 0.0
        else:
            d[i] = 100.0 * (x[i] - prev) / prev
    return d

def delta_points(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        d[i] = x[i] - x[i-1]
    return d

def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = float(np.mean(x[lo:i+1]))
    return out

def compute_te_v(te_T: np.ndarray):
    # E = 1 - T^2 (Lorentz-like reserve)
    E = 1.0 - np.square(te_T)
    vT = np.zeros_like(te_T, dtype=float)
    vE = np.zeros_like(E, dtype=float)
    for i in range(1, len(te_T)):
        vT[i] = te_T[i] - te_T[i-1]
        vE[i] = E[i] - E[i-1]
    return E, vT, vE

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def compute_dn(hrv: np.ndarray, rr: np.ndarray, spo2: np.ndarray, w: int):
    """
    DN_dynamic baseline-free:
    - HRV uses %Δ (bad when negative)
    - RR uses %Δ (bad when positive)
    - SpO2 uses Δ points (bad when negative)
    Fusion = mean-active hazards, only if >=2 systems active at that step.
    Also returns step emergency flags and rolling trend.
    """
    # --- deltas
    d_hrv_pct = pct_change(hrv)
    d_rr_pct  = pct_change(rr)
    d_spo2_pt = delta_points(spo2)

    # --- TT (signed)
    T_hrv = d_hrv_pct / K_HRV
    T_rr  = d_rr_pct  / K_RR
    T_spo2 = d_spo2_pt / K_SPO2

    # --- E, vT, vE
    E_hrv, vT_hrv, vE_hrv = compute_te_v(T_hrv)
    E_rr,  vT_rr,  vE_rr  = compute_te_v(T_rr)
    E_spo2,vT_spo2,vE_spo2= compute_te_v(T_spo2)

    # --- Directional hazards (only deterioration direction counts)
    # HRV: drop (T negative) is bad
    h_hrv  = np.array([clamp01(max(0.0, -t)) for t in T_hrv], dtype=float)
    # RR: rise (T positive) is bad
    h_rr   = np.array([clamp01(max(0.0,  t)) for t in T_rr], dtype=float)
    # SpO2: drop (T negative) is bad
    h_spo2 = np.array([clamp01(max(0.0, -t)) for t in T_spo2], dtype=float)

    # --- Mean-active fusion (>=2 systems active)
    dn_step = np.zeros(len(hrv), dtype=float)
    active_count = np.zeros(len(hrv), dtype=int)
    for i in range(len(hrv)):
        hs = []
        if h_hrv[i]  > 0: hs.append(h_hrv[i])
        if h_rr[i]   > 0: hs.append(h_rr[i])
        if h_spo2[i] > 0: hs.append(h_spo2[i])
        active_count[i] = len(hs)
        if len(hs) >= 2:
            dn_step[i] = float(np.mean(hs))
        else:
            dn_step[i] = 0.0

    dn_roll = rolling_mean(dn_step, w=w)

    # --- STEP emergency flags (hard + fusion)
    step_red = np.zeros(len(hrv), dtype=bool)
    step_warn = np.zeros(len(hrv), dtype=bool)

    for i in range(len(hrv)):
        hard_red = (
            (d_hrv_pct[i] <= HRV_STEP_RED_PCT) or
            (d_rr_pct[i]  >= RR_STEP_RED_PCT)  or
            (d_spo2_pt[i] <= SPO2_STEP_RED_PT)
        )
        soft_red = dn_step[i] >= DN_STEP_RED
        soft_warn = dn_step[i] >= DN_STEP_WARN

        step_red[i] = bool(hard_red or soft_red)
        step_warn[i] = bool((not step_red[i]) and soft_warn)

    # --- Labels: STEP overrides ROLL
    label = np.array(["GREEN"] * len(hrv), dtype=object)

    # rolling trend
    label[dn_roll >= DN_ROLL_WARN] = "WARNING"
    label[dn_roll >= DN_ROLL_RED]  = "RED"

    # step override
    label[step_warn] = "WARNING"
    label[step_red]  = "RED"

    # --- Overall status at last point
    last = len(hrv) - 1
    overall = "STABLE"
    msg = "No significant multi-system reserve acceleration detected."
    if label[last] == "WARNING":
        overall = "WARNING"
        msg = "Multi-system deterioration detected (step and/or rolling)."
    if label[last] == "RED":
        overall = "PRE-FAILURE ALERT"
        msg = "Sustained drift and/or step emergency detected."

    df = pd.DataFrame({
        "min": np.arange(1, len(hrv) + 1),
        "HRV": hrv,
        "RR": rr,
        "SpO2": spo2,

        "dHRV_%": d_hrv_pct,
        "dRR_%": d_rr_pct,
        "dSpO2_pt": d_spo2_pt,

        "T_HRV": T_hrv,
        "E_HRV": E_hrv,
        "vT_HRV": vT_hrv,
        "vE_HRV": vE_hrv,

        "T_RR": T_rr,
        "E_RR": E_rr,
        "vT_RR": vT_rr,
        "vE_RR": vE_rr,

        "T_SpO2": T_spo2,
        "E_SpO2": E_spo2,
        "vT_SpO2": vT_spo2,
        "vE_SpO2": vE_spo2,

        "h_HRV": h_hrv,
        "h_RR": h_rr,
        "h_SpO2": h_spo2,
        "active_n": active_count,

        "DN_step": dn_step,
        "DN_roll": dn_roll,
        "STEP_RED": step_red,
        "STEP_WARN": step_warn,
        "label": label
    })

    return df, overall, msg

def plot_dn(df: pd.DataFrame, title: str):
    x = df["min"].values
    y = df["DN_roll"].values
    labels = df["label"].values

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, y, linewidth=2)

    # colored points
    for xi, yi, lab in zip(x, y, labels):
        if lab == "RED":
            ax.scatter([xi], [yi], s=45, marker="o")
        elif lab == "WARNING":
            ax.scatter([xi], [yi], s=45, marker="o")
        else:
            ax.scatter([xi], [yi], s=35, marker="o")

    ax.axhline(DN_ROLL_WARN, linestyle="--", linewidth=1)
    ax.axhline(DN_ROLL_RED,  linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Time (minute index)")
    ax.set_ylabel("DN_roll")
    ax.set_xticks(np.arange(1, len(x) + 1, 1))
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25)
    return fig

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="DN alert demo", layout="wide")

st.title("DN alert demo")
st.caption("DN_dynamic (baseline-free): mean-active fusion of HRV · RR · SpO₂ + STEP emergency + ROLLING trend")

# Session state for play
if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_i" not in st.session_state:
    st.session_state.play_i = N_POINTS

# Two columns layout
colL, colR = st.columns([1.05, 1.4], gap="large")

with colL:
    st.subheader(f"Input ({N_POINTS} points each)")

    default_hrv = "45 44 42 35 36 32 30 28 27 22 24 23 21 20 19"
    default_rr  = "14 14 15 15 16 16 17 18 19 20 21 22 23 24 25"
    default_spo2= "98 97 92 97 95 94 94 92 93 89 90 91 90 89 89"

    hrv_txt = st.text_area("HRV (ms)", value=default_hrv, height=70)
    rr_txt  = st.text_area("RR (breaths/min)", value=default_rr, height=70)
    spo2_txt= st.text_area("SpO₂ (%)", value=default_spo2, height=70)

    # window choice (no "Parameters" section)
    w = st.selectbox("Rolling window (minutes)", options=[3, 5], index=0)

    btn1, btn2, btn3 = st.columns([1, 1, 0.8])
    with btn1:
        run_static = st.button("Run static", use_container_width=True)
    with btn2:
        play = st.button("Play (3s/step)", use_container_width=True)
    with btn3:
        stop = st.button("Stop", use_container_width=True)

with colR:
    status_box = st.empty()
    chart_box = st.empty()
    details_box = st.empty()

# Parse inputs
hrv = parse_series(hrv_txt, N_POINTS)
rr  = parse_series(rr_txt, N_POINTS)
spo2= parse_series(spo2_txt, N_POINTS)

df, overall, msg = compute_dn(hrv, rr, spo2, w=w)

def render(view_n: int):
    view_df = df.iloc[:view_n].copy()

    # Status based on last visible point
    last_lab = view_df["label"].iloc[-1]
    if last_lab == "RED":
        header = "🔴 PRE-FAILURE ALERT"
        sub = "Step emergency and/or rolling deterioration detected."
    elif last_lab == "WARNING":
        header = "🟠 WARNING"
        sub = "Multi-system deterioration detected (step and/or rolling)."
    else:
        header = "🟢 STABLE"
        sub = "No significant multi-system deterioration detected."

    status_box.markdown(f"### {header}\n{sub}")

    fig = plot_dn(view_df, title=f"DN dynamic (rolling, window={w} min)")
    chart_box.pyplot(fig, clear_figure=True)

    with details_box.container():
        with st.expander("Details (TT/E/vT/vE + step flags)", expanded=False):
            st.dataframe(view_df, use_container_width=True)

# Button logic
if stop:
    st.session_state.playing = False
    st.session_state.play_i = N_POINTS

if run_static:
    st.session_state.playing = False
    st.session_state.play_i = N_POINTS

if play:
    st.session_state.playing = True
    st.session_state.play_i = 1

# Render frame
if st.session_state.playing:
    i = st.session_state.play_i
    render(i)

    if i < N_POINTS:
        time.sleep(3)  # 3 seconds per step
        st.session_state.play_i = i + 1
        st.experimental_rerun()
    else:
        st.session_state.playing = False
else:
    # Static mode shows all points
    render(N_POINTS)
