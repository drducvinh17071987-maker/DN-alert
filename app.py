import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DN alert demo", layout="wide")

# -------------------------
# Helpers
# -------------------------
def parse_series(text: str) -> list[float]:
    if not text.strip():
        return []
    parts = text.replace(",", " ").split()
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except:
            pass
    return out

def pct_change(x: np.ndarray) -> np.ndarray:
    """
    %Δ between consecutive points: 100*(x[i]-x[i-1])/x[i-1], first = 0.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        denom = x[i-1]
        if denom == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 * (x[i] - x[i-1]) / denom
    return out

def clamp01(a):
    return np.minimum(1.0, np.maximum(0.0, a))

def ewma(series: np.ndarray, alpha: float) -> np.ndarray:
    """
    EWMA smoothing: y[t]=alpha*y[t-1]+(1-alpha)*x[t]
    """
    x = np.asarray(series, dtype=float)
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i-1] + (1 - alpha) * x[i]
    return y

def classify(dn_value: float, thr_warn: float, thr_red: float) -> str:
    if dn_value >= thr_red:
        return "RED"
    if dn_value >= thr_warn:
        return "WARNING"
    return "GREEN"

# -------------------------
# DN core (baseline-free, multi-system)
# -------------------------
def compute_dn_dynamic(hrv, rr, spo2,
                       K_hrv=80.0, K_rr=25.0, K_spo2=5.0,
                       w=3,
                       alpha=0.75,
                       thr_warn=0.22, thr_red=0.35,
                       shock_thr=0.35,
                       min_systems_for_alert=2):
    """
    DN philosophy:
    - Only %Δ is input; DN is not thresholding raw values.
    - Convert each %Δ into a *badness step* (0..1) with proper direction:
        HRV: drop is bad
        RR: increase is bad
        SpO2: drop is bad
    - Require >=2 systems active to reduce false alarms (default).
    - Two signals in parallel:
        (A) SHOCK step alert: sudden multi-system event.
        (B) DRIFT alert: sustained deterioration captured by EWMA + Worst-in-window.
    """

    n = len(hrv)
    hrv = np.asarray(hrv, dtype=float)
    rr = np.asarray(rr, dtype=float)
    spo2 = np.asarray(spo2, dtype=float)

    # %Δ
    d_hrv = pct_change(hrv)     # negative is bad
    d_rr = pct_change(rr)       # positive is bad
    d_spo2 = pct_change(spo2)   # negative is bad

    # Signed "bad direction" (positive means deterioration)
    # HRV: drop => -d_hrv
    # RR : rise => +d_rr
    # SpO2: drop => -d_spo2
    bad_hrv = np.maximum(0.0, (-d_hrv) / K_hrv)
    bad_rr = np.maximum(0.0, ( d_rr) / K_rr)
    bad_spo2 = np.maximum(0.0, (-d_spo2) / K_spo2)

    bad_hrv = clamp01(bad_hrv)
    bad_rr = clamp01(bad_rr)
    bad_spo2 = clamp01(bad_spo2)

    # Active systems per minute (badness > 0)
    active = (bad_hrv > 0).astype(int) + (bad_rr > 0).astype(int) + (bad_spo2 > 0).astype(int)

    # Mean-active fusion (only among active systems), else 0
    mean_active = np.zeros(n, dtype=float)
    for i in range(n):
        vals = []
        if bad_hrv[i] > 0: vals.append(bad_hrv[i])
        if bad_rr[i] > 0: vals.append(bad_rr[i])
        if bad_spo2[i] > 0: vals.append(bad_spo2[i])
        mean_active[i] = float(np.mean(vals)) if len(vals) > 0 else 0.0

    # Multi-system gating (reduce false alarms)
    dn_step = np.where(active >= min_systems_for_alert, mean_active, 0.0)

    # SHOCK: sudden multi-system spike OR very strong SpO2 drop with at least one support sign
    shock = np.zeros(n, dtype=int)
    max_bad = np.maximum.reduce([bad_hrv, bad_rr, bad_spo2])
    support_for_spo2 = ((bad_rr > 0) | (bad_hrv > 0)).astype(int)

    for i in range(n):
        is_multisys_spike = (active[i] >= min_systems_for_alert) and (max_bad[i] >= shock_thr)
        is_spo2_critical = (bad_spo2[i] >= 0.60) and (support_for_spo2[i] == 1)  # anti false-alarm
        shock[i] = 1 if (is_multisys_spike or is_spo2_critical) else 0

    # DRIFT: EWMA on dn_step
    dn_ewma = ewma(dn_step, alpha=alpha)

    # Worst-in-window on dn_step (rolling max, inclusive)
    w = int(max(2, w))
    dn_worst = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - (w - 1))
        dn_worst[i] = float(np.max(dn_step[lo:i+1]))

    # Final DN = max(parallel signals)
    dn_final = np.maximum(dn_ewma, dn_worst)

    # Labels per minute
    labels = [classify(dn_final[i], thr_warn, thr_red) for i in range(n)]

    # Minutes lists (1-indexed minutes)
    red_minutes = [i+1 for i, lb in enumerate(labels) if lb == "RED"]
    warn_minutes = [i+1 for i, lb in enumerate(labels) if lb == "WARNING"]

    # Also show shock minutes explicitly (can be RED even if drift low)
    shock_minutes = [i+1 for i in range(n) if shock[i] == 1]

    # Compute "T/E" per system for vt/ve visualization (DN spirit: dynamics)
    # Here T_dyn = signed deterioration (not absolute):
    T_hrv = (-d_hrv) / K_hrv
    T_rr = ( d_rr) / K_rr
    T_spo2 = (-d_spo2) / K_spo2

    # Lorentz-like E (for visualization only)
    E_hrv = 1.0 - np.square(T_hrv)
    E_rr = 1.0 - np.square(T_rr)
    E_spo2 = 1.0 - np.square(T_spo2)

    vT_hrv = np.r_[0.0, np.diff(T_hrv)]
    vT_rr = np.r_[0.0, np.diff(T_rr)]
    vT_spo2 = np.r_[0.0, np.diff(T_spo2)]

    vE_hrv = np.r_[0.0, np.diff(E_hrv)]
    vE_rr = np.r_[0.0, np.diff(E_rr)]
    vE_spo2 = np.r_[0.0, np.diff(E_spo2)]

    df = pd.DataFrame({
        "min": np.arange(1, n+1),
        "HRV": hrv, "RR": rr, "SpO2": spo2,
        "%dHRV": d_hrv, "%dRR": d_rr, "%dSpO2": d_spo2,
        "bad_HRV": bad_hrv, "bad_RR": bad_rr, "bad_SpO2": bad_spo2,
        "active_systems": active,
        "dn_step(mean_active,gated)": dn_step,
        "dn_worst(window)": dn_worst,
        "dn_ewma(drift)": dn_ewma,
        "DN_final": dn_final,
        "label": labels,
        "shock": shock,
        "T_hrv": T_hrv, "T_rr": T_rr, "T_spo2": T_spo2,
        "E_hrv": E_hrv, "E_rr": E_rr, "E_spo2": E_spo2,
        "vT_hrv": vT_hrv, "vT_rr": vT_rr, "vT_spo2": vT_spo2,
        "vE_hrv": vE_hrv, "vE_rr": vE_rr, "vE_spo2": vE_spo2,
    })

    return dn_step, dn_worst, dn_ewma, dn_final, labels, red_minutes, warn_minutes, shock_minutes, df

def build_dn_plot(dn_final, labels, thr_warn, thr_red, title="DN dynamic (parallel: drift + worst)"):
    n = len(dn_final)
    x = np.arange(1, n+1)

    # marker colors
    colors = []
    for lb in labels:
        if lb == "RED":
            colors.append("#d62728")
        elif lb == "WARNING":
            colors.append("#ff7f0e")
        else:
            colors.append("#2ca02c")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=dn_final,
        mode="lines+markers",
        marker=dict(size=9, color=colors),
        line=dict(width=2),
        name="DN"
    ))

    # Threshold lines
    fig.add_hline(y=thr_red, line_dash="dash", opacity=0.5)
    fig.add_hline(y=thr_warn, line_dash="dash", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Time (minute index)",
        yaxis_title="DN (0..1)",
        yaxis=dict(range=[0, 1.0]),
        height=430,
        margin=dict(l=40, r=20, t=55, b=40)
    )
    fig.update_xaxes(dtick=1)
    return fig

# -------------------------
# UI
# -------------------------
st.title("DN alert demo")
st.caption("DN_dynamic (baseline-free): parallel signals = SHOCK step + DRIFT (EWMA) + Worst-in-window. Designed for low false alarms.")

colL, colR = st.columns([1, 1.2], gap="large")

with colL:
    st.subheader("Input (15 points each)")
    default_hrv = "45 44 42 35 36 32 30 31 32 35 38 40 42 43 45"
    default_rr = "14 14 15 15 16 16 17 18 19 16 17 18 19 20 21"
    default_spo2 = "98 97 92 97 95 94 94 92 93 94 95 96 97 98 99"

    hrv_txt = st.text_area("HRV (ms)", value=default_hrv, height=80)
    rr_txt = st.text_area("RR (breaths/min)", value=default_rr, height=80)
    spo2_txt = st.text_area("SpO₂ (%)", value=default_spo2, height=80)

    with st.expander("Settings (hidden by default)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            w = st.selectbox("Worst-in-window (minutes)", [3, 5], index=0)
            alpha = st.slider("EWMA alpha (drift smoothing)", 0.50, 0.90, 0.75, 0.05)
            min_sys = st.selectbox("Min active systems (anti false alarm)", [1, 2, 3], index=1)
        with c2:
            thr_warn = st.slider("WARNING threshold (DN)", 0.10, 0.40, 0.22, 0.01)
            thr_red = st.slider("RED threshold (DN)", 0.20, 0.70, 0.35, 0.01)
            shock_thr = st.slider("SHOCK threshold (max badness)", 0.20, 0.80, 0.35, 0.01)

        st.markdown("**Normalization constants (fixed DN cores):**")
        K_hrv = st.number_input("K_hrv", value=80.0, step=1.0)
        K_rr = st.number_input("K_rr", value=25.0, step=1.0)
        K_spo2 = st.number_input("K_spo2", value=5.0, step=0.5)

    run_static = st.button("Run static", use_container_width=True)
    play = st.button("Play (3s/step)", use_container_width=True)
    stop = st.button("Stop", use_container_width=True)

with colR:
    st.subheader("Output")

# Playback state
if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_idx" not in st.session_state:
    st.session_state.play_idx = 1

if stop:
    st.session_state.playing = False
    st.session_state.play_idx = 1

# Parse inputs
hrv = parse_series(hrv_txt)
rr = parse_series(rr_txt)
spo2 = parse_series(spo2_txt)

ok = (len(hrv) == len(rr) == len(spo2) == 15)

if not ok:
    with colR:
        st.error("Bạn phải nhập đúng **15 số** cho mỗi dãy (HRV, RR, SpO₂).")
    st.stop()

# Compute DN for full series once
dn_step, dn_worst, dn_ewma, dn_final, labels, red_minutes, warn_minutes, shock_minutes, df = compute_dn_dynamic(
    hrv, rr, spo2,
    K_hrv=K_hrv, K_rr=K_rr, K_spo2=K_spo2,
    w=w, alpha=alpha,
    thr_warn=thr_warn, thr_red=thr_red,
    shock_thr=shock_thr,
    min_systems_for_alert=min_sys
)

def render(min_upto: int):
    # slice to minute
    dn_slice = dn_final[:min_upto]
    labels_slice = labels[:min_upto]

    # per-minute label text
    red_m = [m for m in red_minutes if m <= min_upto]
    warn_m = [m for m in warn_minutes if m <= min_upto]
    shock_m = [m for m in shock_minutes if m <= min_upto]

    # Overall status at current minute
    current_label = labels[min_upto - 1]
    if current_label == "RED":
        st.error(f"🔴 RED (minute {min_upto})")
    elif current_label == "WARNING":
        st.warning(f"🟠 WARNING (minute {min_upto})")
    else:
        st.success(f"🟢 STABLE (minute {min_upto})")

    st.markdown(
        f"**RED minutes:** {red_m if red_m else 'None'}  \n"
        f"**WARNING minutes:** {warn_m if warn_m else 'None'}  \n"
        f"**SHOCK minutes (step events):** {shock_m if shock_m else 'None'}"
    )

    fig = build_dn_plot(
        dn_final, labels,
        thr_warn=thr_warn, thr_red=thr_red,
        title=f"DN dynamic — parallel (Worst-in-window={w}m, EWMA α={alpha:.2f})"
    )
    # show full plot but it's okay; minute focus is from labels list above
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Details (per minute: %Δ, T/E, vT/vE, badness, DN parts)", expanded=False):
        st.dataframe(df, use_container_width=True, height=320)

with colR:
    placeholder = st.empty()

# Run static
if run_static:
    with colR:
        placeholder.container()
        render(15)

# Play simulation (blocking loop but simplest & stable)
if play:
    st.session_state.playing = True
    st.session_state.play_idx = 1

if st.session_state.playing and not stop:
    with colR:
        for i in range(st.session_state.play_idx, 16):
            placeholder.container()
            render(i)
            st.session_state.play_idx = i + 1
            time.sleep(3)
        st.session_state.playing = False
        st.session_state.play_idx = 1

# Default first view
if (not run_static) and (not play) and (not st.session_state.playing):
    with colR:
        placeholder.container()
        render(15)
