import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ====== CHỐT K ======
K_HRV = 80      # for %ΔHRV (drop)
K_RR  = 25      # for %ΔRR (rise)  (chốt như bạn)
K_SPO2 = 5      # for SpO2 absolute drop (points)  (chốt k=5)

N_POINTS = 30
STEP_SECONDS = 3

# ====== CORE SETTINGS ======
W = 6  # rolling window length (minutes) to capture "drift / sụp dần"
# threshold for "this channel is active" in the rolling window
THR_ACTIVE = 0.18
# DN thresholds for color
THR_AMBER = 0.28
THR_RED   = 0.55

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def parse_series(txt: str):
    vals = [v for v in txt.replace(",", " ").split() if v.strip() != ""]
    return list(map(float, vals))

def pct_change(x):
    x = np.array(x, dtype=float)
    pct = np.zeros(len(x))
    for i in range(1, len(x)):
        if x[i - 1] != 0:
            pct[i] = 100 * (x[i] - x[i - 1]) / x[i - 1]
    return pct

def rolling_sum(arr, w):
    out = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        s = max(0, i - w + 1)
        out[i] = float(np.sum(arr[s:i+1]))
    return out

def dn_dynamic(hrv, rr, spo2):
    """
    DN_dynamic ICU-friendly:
    - HRV: only DROP matters -> max(0, -%ΔHRV)/K_HRV
    - RR: only RISE matters -> max(0, +%ΔRR)/K_RR
    - SpO2: absolute DROP in points -> max(0, prev-curr)/K_SPO2
    - Rolling window sum (W) to capture drift/sụp dần
    - False-alarm filter: activate only if >=2 systems active in window
    """

    hrv = np.array(hrv, dtype=float)
    rr = np.array(rr, dtype=float)
    spo2 = np.array(spo2, dtype=float)

    # step dynamics
    pct_hrv = pct_change(hrv)
    pct_rr  = pct_change(rr)

    hrv_step = np.maximum(0.0, -pct_hrv) / K_HRV
    rr_step  = np.maximum(0.0,  pct_rr)  / K_RR

    spo2_step = np.zeros(len(spo2), dtype=float)
    for i in range(1, len(spo2)):
        drop = max(0.0, spo2[i-1] - spo2[i])  # absolute points
        spo2_step[i] = drop / K_SPO2

    # rolling window "drift" intensity
    hrv_win = rolling_sum(hrv_step, W)
    rr_win  = rolling_sum(rr_step,  W)
    spo2_win = rolling_sum(spo2_step, W)

    dn = np.zeros(len(hrv), dtype=float)

    for i in range(len(hrv)):
        active = sum([
            hrv_win[i]  >= THR_ACTIVE,
            rr_win[i]   >= THR_ACTIVE,
            spo2_win[i] >= THR_ACTIVE
        ])

        if active >= 2:
            dn[i] = max(hrv_win[i], rr_win[i], spo2_win[i])
        else:
            dn[i] = 0.0  # filtered as likely noise/isolated artifact

    return dn, hrv_step, rr_step, spo2_step, hrv_win, rr_win, spo2_win

def dn_state(dn):
    states = []
    for v in dn:
        if v >= THR_RED:
            states.append("RED")
        elif v >= THR_AMBER:
            states.append("AMBER")
        else:
            states.append("GREEN")
    return states

def render_status_box(state_now: str):
    if state_now == "RED":
        st.error("🔴 PRE-FAILURE ALERT (DN_dynamic)")
        st.caption("Sustained multi-system drift detected (rolling window).")
    elif state_now == "AMBER":
        st.warning("🟡 MONITOR CLOSELY (DN_dynamic)")
        st.caption("Multi-system drift rising—watch trend.")
    else:
        st.success("🟢 STABLE (DN_dynamic)")
        st.caption("No sustained multi-system drift detected.")

def dn_plot(dn, states, upto_idx):
    color_map = {"GREEN": "#2ecc71", "AMBER": "#f1c40f", "RED": "#e74c3c"}
    colors = [color_map[s] for s in states[:upto_idx]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, upto_idx + 1)),
        y=dn[:upto_idx],
        mode="lines+markers",
        marker=dict(color=colors, size=9),
        line=dict(color="gray", width=2)
    ))
    fig.add_hline(y=THR_AMBER, line_dash="dot", line_color="gray")
    fig.add_hline(y=THR_RED, line_dash="dot", line_color="gray")
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time (minute index)",
        yaxis_title=f"DN_dynamic (rolling W={W})",
        showlegend=False
    )
    return fig

# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.title("DN Dynamic ICU Demo (HRV · RR · SpO₂)")
st.caption("Baseline-free + rolling drift + false-alarm filtering (≥2 systems)")

if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_idx" not in st.session_state:
    st.session_state.play_idx = 1

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Input (30 points each)")

    hrv_txt = st.text_area(
        "HRV (ms)",
        "45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 "
        "29 28 27 26 25 24 23 22 21 20 19 18 17 16"
    )
    rr_txt = st.text_area(
        "RR (breaths/min)",
        "14 14 14 15 15 16 16 17 18 18 19 19 20 20 21 "
        "21 22 22 23 23 24 24 25 26 26 27 27 28 28 29"
    )
    spo2_txt = st.text_area(
        "SpO₂ (%)",
        "98 98 98 97 97 96 96 95 95 94 94 93 93 92 92 "
        "91 91 90 90 89 89 88 88 87 87 86 86 85 85 84"
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        static_btn = st.button("Run static analysis")
    with c2:
        play_btn = st.button("▶ Play simulation (3s)")
    with c3:
        stop_btn = st.button("Stop")

    st.caption(f"K: HRV={K_HRV}, RR={K_RR}, SpO₂={K_SPO2} | Rolling W={W} | Need ≥2 systems | Input={N_POINTS} points")

def validate_and_trim(hrv_all, rr_all, spo2_all):
    if len(hrv_all) < N_POINTS or len(rr_all) < N_POINTS or len(spo2_all) < N_POINTS:
        return None, None, None, "Please provide at least 30 values for EACH signal."
    return hrv_all[:N_POINTS], rr_all[:N_POINTS], spo2_all[:N_POINTS], None

if stop_btn:
    st.session_state.playing = False
    st.session_state.play_idx = 1

if play_btn:
    st.session_state.playing = True
    st.session_state.play_idx = 1

with right:
    status_box = st.container()
    chart_box = st.empty()

    try:
        hrv_all = parse_series(hrv_txt)
        rr_all = parse_series(rr_txt)
        spo2_all = parse_series(spo2_txt)
        hrv, rr, spo2, err = validate_and_trim(hrv_all, rr_all, spo2_all)
    except Exception:
        err = "Input format error: please use numbers separated by spaces."
        hrv = rr = spo2 = None

    if err:
        st.error(err)
    else:
        dn, hrv_step, rr_step, spo2_step, hrv_win, rr_win, spo2_win = dn_dynamic(hrv, rr, spo2)
        states = dn_state(dn)

        if static_btn and not st.session_state.playing:
            with status_box:
                render_status_box(states[-1])
            chart_box.plotly_chart(dn_plot(dn, states, upto_idx=N_POINTS), use_container_width=True)

        elif st.session_state.playing:
            idx = int(st.session_state.play_idx)
            idx = max(1, min(idx, N_POINTS))

            with status_box:
                render_status_box(states[idx - 1])

            chart_box.plotly_chart(dn_plot(dn, states, upto_idx=idx), use_container_width=True)

            if idx >= N_POINTS:
                st.session_state.playing = False
                st.session_state.play_idx = 1
            else:
                time.sleep(STEP_SECONDS)
                st.session_state.play_idx = idx + 1
                _rerun()

        else:
            st.info("Press **Run static analysis** or **Play simulation (3s)**.")
            chart_box.plotly_chart(dn_plot(dn, states, upto_idx=1), use_container_width=True)
