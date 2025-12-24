import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# Config
# =========================
st.set_page_config(page_title="DN alert demo", layout="wide")
N = 15
W = 5  # rolling window (minutes)

# Constants (your fixed K)
K_HRV = 80.0
K_RR  = 25.0
K_SPO2 = 5.0

# Step (acute) thresholds (tunable but reasonable for demo + stability)
HRV_STEP_RED_PCT = -40.0   # %ΔHRV <= -40% => acute red
RR_STEP_RED_PCT  = +20.0   # %ΔRR  >= +20% => acute red
SPO2_STEP_RED_PCT = -3.0   # %ΔSpO2 <= -3% => acute red (fast drop)
SPO2_STEP_RED_ABS = -4.0   # absolute drop <= -4 points => acute red

# HRV noise-brake (v1.5 idea)
HRV_NOISE_UP_PCT = +70.0
HRV_NOISE_UP_ABS = +60.0   # ms

# Rolling intensity thresholds (DN dynamic)
DN_WARN = 0.35
DN_RED  = 0.55
HOLD_RED_POINTS = 2  # keep red for 2 points to avoid flicker

# =========================
# Helpers
# =========================
def parse_series(text: str) -> list[float]:
    text = text.replace(",", " ").replace("\n", " ").strip()
    if not text:
        return []
    vals = []
    for tok in text.split():
        try:
            vals.append(float(tok))
        except:
            pass
    return vals

def pct_change(arr: np.ndarray) -> np.ndarray:
    # %Δ[i] = 100*(x[i]-x[i-1])/x[i-1], with first = 0
    out = np.zeros_like(arr, dtype=float)
    for i in range(1, len(arr)):
        prev = arr[i-1]
        if prev == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 * (arr[i] - prev) / prev
    return out

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        out[i] = float(np.mean(x[j0:i+1]))
    return out

def compute_dn(hrv, rr, spo2):
    """
    DN dynamic = baseline-free + multi-system gating.
    - Convert each signal to a "badness severity" in [0,1] from step %Δ (and abs for SpO2).
    - Step-rule catches acute events immediately.
    - Rolling-rule (W=5) catches sustained drift and filters noise.
    """

    hrv = np.array(hrv, dtype=float)
    rr  = np.array(rr, dtype=float)
    spo2 = np.array(spo2, dtype=float)

    # step %Δ
    p_hrv  = pct_change(hrv)       # negative is bad
    p_rr   = pct_change(rr)        # positive is bad
    p_spo2 = pct_change(spo2)      # negative is bad

    # abs step for SpO2
    d_spo2_abs = np.zeros_like(spo2, dtype=float)
    d_spo2_abs[1:] = spo2[1:] - spo2[:-1]  # negative is bad

    # Noise brake for HRV (up-spike very large)
    noise_hrv = np.zeros(len(hrv), dtype=bool)
    for i in range(1, len(hrv)):
        if (p_hrv[i] >= HRV_NOISE_UP_PCT) or ((hrv[i] - hrv[i-1]) >= HRV_NOISE_UP_ABS):
            noise_hrv[i] = True

    # Severity in [0,1] (how strongly "reserve is collapsing" this minute)
    # For HRV: drop => bad => severity from (-%Δ)/K_HRV
    sev_hrv = np.array([clamp01((-p) / K_HRV) for p in p_hrv], dtype=float)

    # For RR: rise => bad => severity from (+%Δ)/K_RR
    sev_rr = np.array([clamp01((p) / K_RR) for p in p_rr], dtype=float)

    # For SpO2: drop => bad => severity from (-%Δ)/K_SPO2, plus abs-drop support
    sev_spo2_pct = np.array([clamp01((-p) / K_SPO2) for p in p_spo2], dtype=float)
    sev_spo2_abs = np.array([clamp01((-(d)) / 5.0) for d in d_spo2_abs], dtype=float)  # 5-point drop => ~1
    sev_spo2 = np.maximum(sev_spo2_pct, sev_spo2_abs)

    # If HRV step looks like sensor-noise up-spike, we dampen its severity for that point
    sev_hrv = np.where(noise_hrv, 0.0, sev_hrv)

    # Step-rule flags (acute)
    step_red_hrv  = (p_hrv <= HRV_STEP_RED_PCT) & (~noise_hrv)
    step_red_rr   = (p_rr  >= RR_STEP_RED_PCT)
    step_red_spo2 = (p_spo2 <= SPO2_STEP_RED_PCT) | (d_spo2_abs <= SPO2_STEP_RED_ABS)

    # Multi-system gating: consider "systems in trouble" this minute
    # system is "active" if severity >= 0.25 (light trouble)
    active_hrv  = sev_hrv  >= 0.25
    active_rr   = sev_rr   >= 0.25
    active_spo2 = sev_spo2 >= 0.25

    active_count = active_hrv.astype(int) + active_rr.astype(int) + active_spo2.astype(int)

    # Rolling drift intensity: average of top-2 severities, but only if ≥2 systems active
    sev_stack = np.vstack([sev_hrv, sev_rr, sev_spo2]).T  # shape (N,3)
    top2_mean = np.array([np.mean(np.sort(row)[-2:]) for row in sev_stack], dtype=float)
    dn_raw = np.where(active_count >= 2, top2_mean, 0.0)

    # Rolling window smooth
    dn_roll = rolling_mean(dn_raw, W)

    # Classification per point
    label = np.array(["GREEN"] * len(dn_roll), dtype=object)

    # INFO: HRV noise points
    label = np.where(noise_hrv, "INFO", label)

    # WARNING / RED by rolling intensity
    label = np.where((dn_roll >= DN_WARN) & (active_count >= 2), "WARNING", label)
    label = np.where((dn_roll >= DN_RED)  & (active_count >= 2), "RED", label)

    # Step-rule can force RED if multi-system acute, OR single-system very strong + another active
    step_any = step_red_hrv.astype(int) + step_red_rr.astype(int) + step_red_spo2.astype(int)

    # If 2+ acute flags => RED
    label = np.where(step_any >= 2, "RED", label)

    # If 1 acute flag AND at least one other system active => RED
    label = np.where((step_any == 1) & (active_count >= 2), "RED", label)

    # Hold RED for a few points to reduce flicker
    red_hold = np.zeros(len(label), dtype=int)
    hold = 0
    for i in range(len(label)):
        if label[i] == "RED":
            hold = HOLD_RED_POINTS
            red_hold[i] = 1
        else:
            if hold > 0:
                label[i] = "RED"
                red_hold[i] = 1
                hold -= 1

    # Provide a compact summary (final status)
    final_status = "STABLE"
    if np.any(label == "RED"):
        final_status = "PRE-FAILURE ALERT"
    elif np.any(label == "WARNING"):
        final_status = "WARNING"
    elif np.any(label == "INFO"):
        final_status = "INFO"

    df = pd.DataFrame({
        "t(min)": np.arange(1, len(hrv)+1),
        "HRV": hrv,
        "RR": rr,
        "SpO2": spo2,
        "%dHRV": np.round(p_hrv, 2),
        "%dRR": np.round(p_rr, 2),
        "%dSpO2": np.round(p_spo2, 2),
        "sev_HRV": np.round(sev_hrv, 3),
        "sev_RR": np.round(sev_rr, 3),
        "sev_SpO2": np.round(sev_spo2, 3),
        "DN_raw": np.round(dn_raw, 3),
        "DN_roll": np.round(dn_roll, 3),
        "label": label
    })

    return df, final_status

def make_plot(df: pd.DataFrame, title_suffix="(static)"):
    x = df["t(min)"].tolist()
    y = df["DN_roll"].tolist()
    labels = df["label"].tolist()

    # color map
    c = []
    for lb in labels:
        if lb == "RED":
            c.append("#d62728")
        elif lb == "WARNING":
            c.append("#ff7f0e")
        elif lb == "INFO":
            c.append("#1f77b4")
        else:
            c.append("#2ca02c")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        marker=dict(size=9, color=c),
        line=dict(width=2, color="#111111"),
        name="DN dynamic"
    ))

    # add threshold lines
    fig.add_hline(y=DN_WARN, line_dash="dot")
    fig.add_hline(y=DN_RED,  line_dash="dot")

    fig.update_layout(
        title=f"DN dynamic {title_suffix}",
        height=520,
        margin=dict(l=40, r=20, t=60, b=50),
        xaxis_title="Time (minute index)",
        yaxis_title="DN dynamic (rolling 5-min)",
        showlegend=False
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x,
        ticktext=[str(i) for i in x],
        range=[1, N]
    )
    fig.update_yaxes(range=[0, 1.0])
    return fig

# =========================
# UI
# =========================
st.title("DN alert demo")
st.caption("Baseline-free + rolling drift + false-alarm filtering (≥2 systems)")

colL, colR = st.columns([1.05, 1.25], gap="large")

with colL:
    st.subheader("Input (15 points each)")

    hrv_text = st.text_area(
        "HRV (ms)",
        value="45 44 42 35 36 32 30 28 27 22 24 23 21 20 19",
        height=90
    )
    rr_text = st.text_area(
        "RR (breaths/min)",
        value="14 14 15 15 16 16 17 18 19 20 21 22 23 24 25",
        height=90
    )
    spo2_text = st.text_area(
        "SpO₂ (%)",
        value="98 97 92 97 95 94 94 92 93 89 90 90 91 90 89",
        height=90
    )

    c1, c2, c3 = st.columns([1,1,1])
    run_static = c1.button("Run static", use_container_width=True)
    play = c2.button("▶ Play (3s/step)", use_container_width=True)
    stop = c3.button("Stop", use_container_width=True)

    if "stop_flag" not in st.session_state:
        st.session_state.stop_flag = False
    if stop:
        st.session_state.stop_flag = True
    if play:
        st.session_state.stop_flag = False

with colR:
    status_box = st.empty()
    chart_box = st.empty()
    table_box = st.empty()

def get_inputs_or_stop():
    hrv = parse_series(hrv_text)[:N]
    rr  = parse_series(rr_text)[:N]
    spo2= parse_series(spo2_text)[:N]

    if len(hrv) < N or len(rr) < N or len(spo2) < N:
        st.error("Please input exactly 15 numbers for each signal.")
        st.stop()
    return hrv, rr, spo2

def render_static():
    hrv, rr, spo2 = get_inputs_or_stop()
    df, final_status = compute_dn(hrv, rr, spo2)

    if final_status == "PRE-FAILURE ALERT":
        status_box.error(f"● {final_status} (DN_dynamic)")
    elif final_status == "WARNING":
        status_box.warning(f"● {final_status} (DN_dynamic)")
    elif final_status == "INFO":
        status_box.info(f"● {final_status} (DN_dynamic)")
    else:
        status_box.success(f"● {final_status} (DN_dynamic)")

    fig = make_plot(df, "(static)")
    chart_box.plotly_chart(fig, use_container_width=True)

    # compact table (optional, still useful)
    show_cols = ["t(min)", "HRV", "RR", "SpO2", "%dHRV", "%dRR", "%dSpO2", "DN_roll", "label"]
    table_box.dataframe(df[show_cols], use_container_width=True, height=260)

def render_play():
    hrv, rr, spo2 = get_inputs_or_stop()
    df_full, _ = compute_dn(hrv, rr, spo2)

    for k in range(1, N+1):
        if st.session_state.stop_flag:
            break

        df = df_full.iloc[:k].copy()

        # status based on visible window
        labels = df["label"].values
        if np.any(labels == "RED"):
            status_box.error("● PRE-FAILURE ALERT (DN_dynamic)")
        elif np.any(labels == "WARNING"):
            status_box.warning("● WARNING (DN_dynamic)")
        elif np.any(labels == "INFO"):
            status_box.info("● INFO (DN_dynamic)")
        else:
            status_box.success("● STABLE (DN_dynamic)")

        fig = make_plot(df, f"(play: 1→{k})")
        chart_box.plotly_chart(fig, use_container_width=True)

        time.sleep(3)

# Auto-run
if run_static:
    render_static()
elif play:
    render_play()
else:
    # default: show static once at load (nice for video)
    render_static()
