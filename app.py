import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# Fixed constants (no Parameters panel)
# ----------------------------
N_POINTS = 15

K_HRV = 80.0          # DN_dynamic HRV constant (pct)
K_RR = 25.0           # DN_dynamic RR constant (pct)
K_SPO2 = 5.0          # DN_dynamic SpO2 constant (points)

ROLL_W = 5            # rolling window (set 3 or 5)
PLAY_SECONDS = 3

# thresholds (visual + label)
TH_WARN = 0.25
TH_RED = 0.55

# minimal false-alarm filters
HRV_NOISE_PCT_UP = 70.0
HRV_NOISE_ABS_UP = 60.0

SPO2_NOISE_DROP = -4.0   # points
SPO2_NOISE_RECOV = +4.0  # points

EPS_ACTIVE = 1e-9


# ----------------------------
# Helpers
# ----------------------------
def parse_series(s: str) -> list[float]:
    s = (s or "").replace(",", " ").strip()
    if not s:
        return []
    parts = [p for p in s.split() if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except:
            pass
    return out


def pct_delta(x: np.ndarray) -> np.ndarray:
    """%Δ between consecutive points; first = 0."""
    d = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        prev = x[i - 1]
        if abs(prev) < 1e-12:
            d[i] = 0.0
        else:
            d[i] = 100.0 * (x[i] - prev) / prev
    return d


def pt_delta(x: np.ndarray) -> np.ndarray:
    """point delta; first = 0."""
    d = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        d[i] = x[i] - x[i - 1]
    return d


def clip_t(T: np.ndarray) -> np.ndarray:
    """Keep Lorentz stable. Baseline-free demo: clip to [-1, 1]."""
    return np.clip(T, -1.0, 1.0)


def lorentz_E(T: np.ndarray) -> np.ndarray:
    return 1.0 - (T ** 2)


def velocity(arr: np.ndarray) -> np.ndarray:
    v = np.zeros_like(arr, dtype=float)
    for i in range(1, len(arr)):
        v[i] = arr[i] - arr[i - 1]
    return v


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        a = max(0, i - w + 1)
        out[i] = float(np.mean(x[a:i+1]))
    return out


def compute_dn_mean_active(hrv, rr, spo2):
    """
    DN_dynamic (baseline-free):
    - HRV uses %ΔHRV / K_HRV
    - RR uses  %ΔRR  / K_RR
    - SpO2 uses ΔSpO2(points)/ K_SPO2
    - Active = only worsening direction:
        HRV drop => -T_hrv
        RR  rise => +T_rr
        SpO2 drop=> -T_spo2
    - DN_step = mean(active scores)
    - DN_roll = rolling mean
    Also returns per-system T/E/vT/vE and mean-active T/E/vT/vE.
    """

    n = len(hrv)

    # --- deltas
    d_hrv_pct = pct_delta(hrv)
    d_rr_pct = pct_delta(rr)
    d_spo2_pt = pt_delta(spo2)

    # --- T per system (dynamic)
    T_hrv = clip_t(d_hrv_pct / K_HRV)
    T_rr = clip_t(d_rr_pct / K_RR)
    T_spo2 = clip_t(d_spo2_pt / K_SPO2)

    # --- E per system
    E_hrv = lorentz_E(T_hrv)
    E_rr = lorentz_E(T_rr)
    E_spo2 = lorentz_E(T_spo2)

    # --- velocities per system
    vT_hrv, vE_hrv = velocity(T_hrv), velocity(E_hrv)
    vT_rr,  vE_rr  = velocity(T_rr),  velocity(E_rr)
    vT_spo2,vE_spo2= velocity(T_spo2),velocity(E_spo2)

    # ----------------------------
    # Minimal false-alarm filtering (step-level)
    # ----------------------------

    # HRV noise: very fast rise can be sensor/position artifact -> ignore that step’s HRV worsening score
    hrv_abs_step = np.zeros(n, dtype=float)
    for i in range(1, n):
        hrv_abs_step[i] = hrv[i] - hrv[i-1]
    hrv_noise_step = (d_hrv_pct >= HRV_NOISE_PCT_UP) | (hrv_abs_step >= HRV_NOISE_ABS_UP)

    # SpO2 noise: dip then immediate recover -> ignore dip step
    spo2_noise_step = np.zeros(n, dtype=bool)
    for i in range(1, n-1):
        if d_spo2_pt[i] <= SPO2_NOISE_DROP and d_spo2_pt[i+1] >= SPO2_NOISE_RECOV:
            spo2_noise_step[i] = True

    # ----------------------------
    # Scores (only “worsening” direction)
    # ----------------------------
    # HRV worsening: drop => negative T_hrv => score = max(0, -T_hrv)
    score_hrv = np.maximum(0.0, -T_hrv)
    score_hrv[hrv_noise_step] = 0.0

    # RR worsening: rise => positive T_rr => score = max(0, +T_rr)
    score_rr = np.maximum(0.0, T_rr)

    # SpO2 worsening: drop => negative T_spo2 => score = max(0, -T_spo2)
    score_spo2 = np.maximum(0.0, -T_spo2)
    score_spo2[spo2_noise_step] = 0.0

    # ----------------------------
    # DN_step = mean of active systems
    # ----------------------------
    dn_step = np.zeros(n, dtype=float)
    active_count = np.zeros(n, dtype=int)

    for i in range(n):
        scores = []
        if score_hrv[i] > EPS_ACTIVE: scores.append(score_hrv[i])
        if score_rr[i]  > EPS_ACTIVE: scores.append(score_rr[i])
        if score_spo2[i]> EPS_ACTIVE: scores.append(score_spo2[i])

        active_count[i] = len(scores)
        dn_step[i] = float(np.mean(scores)) if scores else 0.0

    dn_roll = rolling_mean(dn_step, ROLL_W)

    # ----------------------------
    # mean-active T/E and their velocities (aggregated)
    # Here we aggregate T in the SAME worsening sign space:
    #   T_eff_hrv = -T_hrv (only when HRV worsening)
    #   T_eff_rr  = +T_rr  (only when RR worsening)
    #   T_eff_spo2= -T_spo2(only when SpO2 worsening)
    # And mean over active systems.
    # ----------------------------
    T_eff_mean = np.zeros(n, dtype=float)
    for i in range(n):
        Ts = []
        if score_hrv[i] > EPS_ACTIVE: Ts.append(-T_hrv[i])
        if score_rr[i]  > EPS_ACTIVE: Ts.append(T_rr[i])
        if score_spo2[i]> EPS_ACTIVE: Ts.append(-T_spo2[i])
        T_eff_mean[i] = float(np.mean(Ts)) if Ts else 0.0

    T_eff_mean = clip_t(T_eff_mean)
    E_eff_mean = lorentz_E(T_eff_mean)
    vT_eff_mean = velocity(T_eff_mean)
    vE_eff_mean = velocity(E_eff_mean)

    # ----------------------------
    # labels per timepoint (from dn_roll)
    # ----------------------------
    label = np.array(["GREEN"] * n, dtype=object)
    label[dn_roll >= TH_WARN] = "WARNING"
    label[dn_roll >= TH_RED] = "RED"

    return {
        "d_hrv_pct": d_hrv_pct, "d_rr_pct": d_rr_pct, "d_spo2_pt": d_spo2_pt,
        "T_hrv": T_hrv, "E_hrv": E_hrv, "vT_hrv": vT_hrv, "vE_hrv": vE_hrv,
        "T_rr": T_rr, "E_rr": E_rr, "vT_rr": vT_rr, "vE_rr": vE_rr,
        "T_spo2": T_spo2, "E_spo2": E_spo2, "vT_spo2": vT_spo2, "vE_spo2": vE_spo2,
        "score_hrv": score_hrv, "score_rr": score_rr, "score_spo2": score_spo2,
        "active_count": active_count,
        "dn_step": dn_step, "dn_roll": dn_roll,
        "T_eff_mean": T_eff_mean, "E_eff_mean": E_eff_mean,
        "vT_eff_mean": vT_eff_mean, "vE_eff_mean": vE_eff_mean,
        "label": label
    }


def status_from_last(label_last: str) -> tuple[str, str]:
    if label_last == "RED":
        return ("PRE-FAILURE ALERT (DN_dynamic)", "error")
    if label_last == "WARNING":
        return ("WARNING (DN_dynamic)", "warning")
    return ("STABLE", "success")


def make_dn_plot(x, dn_roll, label, title="DN dynamic (rolling)"):
    fig = go.Figure()

    # color by label
    colors = []
    for lb in label:
        if lb == "RED":
            colors.append("red")
        elif lb == "WARNING":
            colors.append("orange")
        else:
            colors.append("green")

    fig.add_trace(go.Scatter(
        x=x, y=dn_roll,
        mode="lines+markers",
        marker=dict(size=8, color=colors),
        line=dict(width=2),
        name="DN_roll"
    ))

    # thresholds
    fig.add_hline(y=TH_WARN, line_dash="dot", opacity=0.6)
    fig.add_hline(y=TH_RED, line_dash="dot", opacity=0.6)

    fig.update_layout(
        title=title,
        xaxis_title="Time (minute index)",
        yaxis_title=f"DN dynamic (rolling w={ROLL_W})",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 1])
    return fig


def make_v_plot(x, vT, vE, title="vT / vE (mean active)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=vT, mode="lines+markers", name="vT"))
    fig.add_trace(go.Scatter(x=x, y=vE, mode="lines+markers", name="vE"))
    fig.update_layout(
        title=title,
        xaxis_title="Time (minute index)",
        yaxis_title="Velocity",
        height=320,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(dtick=1)
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="DN alert demo", layout="wide")
st.title("DN alert demo")
st.caption("DN_dynamic (baseline-free): mean-active fusion of HRV · RR · SpO₂  |  with vT/vE (mean active)")

# init session
if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_i" not in st.session_state:
    st.session_state.play_i = 1

colL, colR = st.columns([1, 1.25], gap="large")

with colL:
    st.subheader(f"Input ({N_POINTS} points each)")

    default_hrv = "45 44 42 35 36 32 30 28 27 22 24 23 21 20 19"
    default_rr = "14 14 15 15 16 16 17 18 19 20 21 22 23 24 25"
    default_spo2 = "98 97 92 97 95 94 94 92 93 89 90 91 90 89 89"

    hrv_txt = st.text_area("HRV (ms)", value=default_hrv, height=90)
    rr_txt = st.text_area("RR (breaths/min)", value=default_rr, height=90)
    spo2_txt = st.text_area("SpO₂ (%)", value=default_spo2, height=90)

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        run_static = st.button("Run static", use_container_width=True)
    with b2:
        play = st.button("▶ Play (3s/step)", use_container_width=True)
    with b3:
        stop = st.button("Stop", use_container_width=True)

    if play:
        st.session_state.playing = True
        st.session_state.play_i = 1

    if stop:
        st.session_state.playing = False

# parse + validate
hrv_list = parse_series(hrv_txt)
rr_list = parse_series(rr_txt)
spo2_list = parse_series(spo2_txt)

def hard_error(msg: str):
    with colR:
        st.error(msg)
    st.stop()

if len(hrv_list) != N_POINTS or len(rr_list) != N_POINTS or len(spo2_list) != N_POINTS:
    with colR:
        st.info(f"Nhập đúng {N_POINTS} số cho mỗi ô (cách nhau bằng dấu cách).")
    st.stop()

hrv = np.array(hrv_list, dtype=float)
rr = np.array(rr_list, dtype=float)
spo2 = np.array(spo2_list, dtype=float)

res = compute_dn_mean_active(hrv, rr, spo2)

# decide how many points to show
if st.session_state.playing:
    i_show = int(st.session_state.play_i)
    i_show = max(1, min(N_POINTS, i_show))
else:
    i_show = N_POINTS

x = np.arange(1, i_show + 1)

dn_roll_show = res["dn_roll"][:i_show]
label_show = res["label"][:i_show]
vT_show = res["vT_eff_mean"][:i_show]
vE_show = res["vE_eff_mean"][:i_show]

last_label = label_show[-1]
status_text, status_kind = status_from_last(last_label)

with colR:
    # status banner
    if status_kind == "error":
        st.error(status_text)
    elif status_kind == "warning":
        st.warning(status_text)
    else:
        st.success(status_text)

    # charts
    fig1 = make_dn_plot(x, dn_roll_show, label_show, title="DN dynamic (rolling) — mean active")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = make_v_plot(x, vT_show, vE_show, title="vT / vE (mean active) — speed of change")
    st.plotly_chart(fig2, use_container_width=True)

    # optional: compact table (top 8 cols)
    df = pd.DataFrame({
        "t": np.arange(1, N_POINTS + 1),
        "HRV": hrv, "RR": rr, "SpO2": spo2,
        "dn_step": np.round(res["dn_step"], 3),
        "dn_roll": np.round(res["dn_roll"], 3),
        "active_n": res["active_count"],
        "T_mean": np.round(res["T_eff_mean"], 3),
        "E_mean": np.round(res["E_eff_mean"], 3),
        "vT_mean": np.round(res["vT_eff_mean"], 3),
        "vE_mean": np.round(res["vE_eff_mean"], 3),
        "label": res["label"]
    })
    st.caption("Table (for checking) — you can hide this later for clean video.")
    st.dataframe(df.head(i_show), use_container_width=True, height=260)

# play engine (auto-advance)
if st.session_state.playing:
    if st.session_state.play_i < N_POINTS:
        time.sleep(PLAY_SECONDS)
        st.session_state.play_i += 1
        st.rerun()
    else:
        st.session_state.playing = False
