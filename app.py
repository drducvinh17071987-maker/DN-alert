import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================
# DN DYNAMIC PARAMETERS (CHỐT)
# =========================
K_HRV = 80
K_RR = 25
K_SPO2 = 5
N_POINTS = 30
STEP_SECONDS = 3  # Play: 3 seconds per step

# =========================
# Helpers
# =========================
def _rerun():
    # Streamlit version compatibility
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

def dn_dynamic(hrv, rr, spo2):
    """
    DN_dynamic (baseline-free):
    - Compute |%Δ| for each signal
    - Normalize by K: HRV/80, RR/25, SpO2/5
    - Filter false alarms: activate only if >=2 systems exceed T>0.25
    """
    hrv = np.array(hrv, dtype=float)
    rr = np.array(rr, dtype=float)
    spo2 = np.array(spo2, dtype=float)

    T_hrv = np.abs(pct_change(hrv)) / K_HRV
    T_rr = np.abs(pct_change(rr)) / K_RR
    T_spo2 = np.abs(pct_change(spo2)) / K_SPO2

    dn = np.zeros(len(hrv))
    for i in range(len(hrv)):
        active = sum([
            T_hrv[i] > 0.25,
            T_rr[i] > 0.25,
            T_spo2[i] > 0.25
        ])
        dn[i] = max(T_hrv[i], T_rr[i], T_spo2[i]) if active >= 2 else 0.0

    return dn

def dn_state(dn):
    """
    Demo thresholds (for visualization only):
    - GREEN: < 0.35
    - AMBER: 0.35–0.6
    - RED: >= 0.6
    """
    states = []
    for v in dn:
        if v >= 0.6:
            states.append("RED")
        elif v >= 0.35:
            states.append("AMBER")
        else:
            states.append("GREEN")
    return states

def render_status_box(state_now: str):
    if state_now == "RED":
        st.error("🔴 PRE-FAILURE ALERT")
        st.caption("Multi-system reserve acceleration detected (DN_dynamic).")
    elif state_now == "AMBER":
        st.warning("🟡 MONITOR CLOSELY")
        st.caption("Reserve is accelerating—watch closely (DN_dynamic).")
    else:
        st.success("🟢 STABLE")
        st.caption("No significant multi-system reserve acceleration detected.")

def dn_plot(dn, states, upto_idx):
    # upto_idx is inclusive count of points to show (1..N_POINTS)
    dn_show = dn[:upto_idx]
    states_show = states[:upto_idx]

    color_map = {"GREEN": "#2ecc71", "AMBER": "#f1c40f", "RED": "#e74c3c"}
    colors = [color_map[s] for s in states_show]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, upto_idx + 1)),
        y=dn_show,
        mode="lines+markers",
        marker=dict(color=colors, size=9),
        line=dict(color="gray", width=2)
    ))

    # reference lines (optional but helpful)
    fig.add_hline(y=0.35, line_dash="dot", line_color="gray")
    fig.add_hline(y=0.6, line_dash="dot", line_color="gray")

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time (minute index)",
        yaxis_title="DN_dynamic intensity",
        showlegend=False
    )
    return fig

# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide")
st.title("DN Prognostic Signal Demo")
st.caption("DN_dynamic (baseline-free) · HRV · RR · SpO₂ · false-alarm filtering (≥2 systems)")

# session state init
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

    st.divider()
    st.caption(f"Chốt K: HRV={K_HRV}, RR={K_RR}, SpO₂={K_SPO2}.  |  Input: {N_POINTS} points.  |  Play: {STEP_SECONDS}s/step")

# Parse + validate once (used by both modes)
try:
    hrv_all = parse_series(hrv_txt)
    rr_all = parse_series(rr_txt)
    spo2_all = parse_series(spo2_txt)
except Exception:
    hrv_all, rr_all, spo2_all = None, None, None

def validate_and_trim():
    if hrv_all is None or rr_all is None or spo2_all is None:
        return None, None, None, "Input format error: please use numbers separated by spaces."

    if len(hrv_all) < N_POINTS or len(rr_all) < N_POINTS or len(spo2_all) < N_POINTS:
        return None, None, None, "Please provide at least 30 values for EACH signal."

    hrv = hrv_all[:N_POINTS]
    rr = rr_all[:N_POINTS]
    spo2 = spo2_all[:N_POINTS]
    return hrv, rr, spo2, None

# Control buttons
if stop_btn:
    st.session_state.playing = False
    st.session_state.play_idx = 1

if play_btn:
    st.session_state.playing = True
    st.session_state.play_idx = 1

# Right side rendering
with right:
    # placeholders so UI doesn't jump
    status_box = st.container()
    chart_box = st.empty()

    hrv, rr, spo2, err = validate_and_trim()

    if err:
        st.error(err)

    else:
        dn = dn_dynamic(hrv, rr, spo2)
        states = dn_state(dn)

        # STATIC MODE
        if static_btn and not st.session_state.playing:
            state_now = states[-1]
            with status_box:
                render_status_box(state_now)

            fig = dn_plot(dn, states, upto_idx=N_POINTS)
            chart_box.plotly_chart(fig, use_container_width=True)

        # PLAY MODE (pseudo-streaming)
        elif st.session_state.playing:
            idx = int(st.session_state.play_idx)
            idx = max(1, min(idx, N_POINTS))

            state_now = states[idx - 1]
            with status_box:
                render_status_box(state_now)

            fig = dn_plot(dn, states, upto_idx=idx)
            chart_box.plotly_chart(fig, use_container_width=True)

            # advance
            if idx >= N_POINTS:
                st.session_state.playing = False
                st.session_state.play_idx = 1
            else:
                time.sleep(STEP_SECONDS)
                st.session_state.play_idx = idx + 1
                _rerun()

        # DEFAULT VIEW (when nothing pressed yet)
        else:
            st.info("Press **Run static analysis** (final view) or **Play simulation (3s)** (streaming view).")
            # show an empty/initial chart with first point for orientation
            fig = dn_plot(dn, states, upto_idx=1)
            chart_box.plotly_chart(fig, use_container_width=True)
