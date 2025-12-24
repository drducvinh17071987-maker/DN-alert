import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DN alert demo", layout="wide")
st.title("DN alert demo")
st.caption("Gatekeeper: DN RED if ≥2 systems are RED; DN WARNING if ≥2 systems are WARNING/RED. Runs only on button clicks.")

# ----------------------------
# Utils
# ----------------------------
def parse_15(text: str):
    text = text.replace(",", " ").replace("\n", " ").strip()
    if not text:
        return []
    vals = []
    for p in text.split():
        try:
            vals.append(float(p))
        except:
            pass
    return vals[:15]

def pct_change(prev, cur):
    if prev == 0:
        return 0.0
    return 100.0 * (cur - prev) / prev

def color_map(label):
    if label == "RED":
        return "#d62728"
    if label == "WARNING":
        return "#ffbf00"
    if label == "INFO":
        return "#1f77b4"
    return "#2ca02c"  # GREEN

def dn_gatekeeper(hrv_label, rr_label, spo2_label):
    # INFO does NOT count as WARNING/RED; it's used for noise/recovery notes.
    labels = [hrv_label, rr_label, spo2_label]
    red_cnt = sum(x == "RED" for x in labels)
    warn_cnt = sum(x in ("RED", "WARNING") for x in labels)

    if red_cnt >= 2:
        return "RED"
    if warn_cnt >= 2:
        return "WARNING"
    return "GREEN"

# ----------------------------
# Per-system rules (your spec)
# ----------------------------
K_SPO2 = 5.0
K_RR = 25.0
K_HRV = 80.0  # dynamic normalization for %ΔHRV if needed

def spo2_label(prev, cur, next_val=None):
    """
    T = ΔSpO2 / 5 (abs drop)
    |T|>=0.6 RED (~3% drop), |T|>=0.3 WARNING (~1.5% drop)
    V-shape (drop then immediate recovery) -> INFO
    """
    drop = max(0.0, prev - cur)           # absolute drop
    T = drop / K_SPO2

    # V-shape noise/recovery check (needs next point)
    if next_val is not None:
        rec = max(0.0, next_val - cur)    # recovery amount
        # if it drops then rebounds quickly close to prior level -> INFO
        if drop >= 2.0 and rec >= 2.0 and abs(next_val - prev) <= 1.0:
            return "INFO", T

    if T >= 0.6:
        return "RED", T
    if T >= 0.3:
        return "WARNING", T
    return "GREEN", T

def rr_label(prev, cur):
    """
    T = %ΔRR / 25
    T>=1 RED (≥25%/min), T>=0.5 WARNING (~12–15%), T>=0.2 INFO (~5–8%)
    """
    d = pct_change(prev, cur)
    up = max(0.0, d)
    T = up / K_RR
    if T >= 1.0:
        return "RED", T
    if T >= 0.5:
        return "WARNING", T
    if T >= 0.2:
        return "INFO", T
    return "GREEN", T

def hrv_label(hrv_series, i):
    """
    HRV uses *shape*, not absolute value.
    - Uses %ΔHRV step, vT, V-shape, drift.
    - Step-drop ≤ -40% => RED (unless immediate V-shape recovery => INFO)
    - V-shape: d1<=-20% and d2>=+15% and |total|<=12% => INFO (noise/recovery)
    - Drift: if last 3 steps total_pct <= -15% => WARNING (soft drift)
    """
    if i == 0:
        return "GREEN", 0.0, 0.0, 0.0  # label, d%, T, E

    prev = hrv_series[i-1]
    cur = hrv_series[i]
    d = pct_change(prev, cur)  # %ΔHRV
    # dynamic T on %ΔHRV (same spirit you used before)
    T = d / K_HRV
    E = 1.0 - (T * T)

    # Step-drop RED rule
    if d <= -40.0:
        # if immediate rebound next step -> treat as INFO (likely artifact)
        if i + 1 < len(hrv_series):
            d2 = pct_change(cur, hrv_series[i+1])
            total = 100.0 * (hrv_series[i+1] - prev) / prev if prev != 0 else 0.0
            if d <= -20.0 and d2 >= +15.0 and abs(total) <= 12.0:
                return "INFO", d, T, E
        return "RED", d, T, E

    # V-shape info (classic)
    if i + 1 < len(hrv_series):
        d2 = pct_change(cur, hrv_series[i+1])
        total = 100.0 * (hrv_series[i+1] - prev) / prev if prev != 0 else 0.0
        if d <= -20.0 and d2 >= +15.0 and abs(total) <= 12.0:
            return "INFO", d, T, E

    # Drift over last 3 steps (soft, low false alarm)
    if i >= 3:
        base = hrv_series[i-3]
        total3 = 100.0 * (cur - base) / base if base != 0 else 0.0
        if total3 <= -15.0:
            return "WARNING", d, T, E

    return "GREEN", d, T, E

# ----------------------------
# Plotting
# ----------------------------
def plot_dn(df, upto):
    d = df.iloc[:upto].copy()
    x = d["minute"].to_list()
    y = d["DN_level"].to_list()
    colors = [color_map(s) for s in d["DN_status"].to_list()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        line=dict(width=3),
        marker=dict(size=10, color=colors, line=dict(width=1, color="#111111")),
        name="DN"
    ))
    fig.update_layout(
        height=430,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Minute",
        yaxis_title="DN (GREEN=0, WARNING=1, RED=2)",
        yaxis=dict(range=[-0.2, 2.2], dtick=1),
    )
    fig.update_xaxes(dtick=1)
    return fig

def plot_raw(df, upto):
    d = df.iloc[:upto].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["minute"], y=d["HRV"], mode="lines+markers", name="HRV"))
    fig.add_trace(go.Scatter(x=d["minute"], y=d["RR"], mode="lines+markers", name="RR"))
    fig.add_trace(go.Scatter(x=d["minute"], y=d["SpO2"], mode="lines+markers", name="SpO2"))
    fig.update_layout(
        height=430,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Minute",
        yaxis_title="Raw values",
    )
    fig.update_xaxes(dtick=1)
    return fig

# ----------------------------
# UI Layout
# ----------------------------
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("Input (15 points)")
    default_hrv = "45 44 42 35 36 32 30 28 27 22 24 23 21 20 19"
    default_rr = "14 14 15 15 16 16 17 18 19 20 21 22 23 24 25"
    default_spo2 = "98 97 92 97 95 94 94 92 93 89 90 91 90 89 89"

    hrv_txt = st.text_area("HRV (ms)", default_hrv, height=80)
    rr_txt = st.text_area("RR (breaths/min)", default_rr, height=80)
    spo2_txt = st.text_area("SpO₂ (%)", default_spo2, height=80)

    cA, cB = st.columns(2)
    with cA:
        static_btn = st.button("Run static", use_container_width=True)
    with cB:
        play_btn = st.button("Play (3s/step)", use_container_width=True)

    st.markdown("**Note:** App only updates when you press a button (no auto-run on edit).")

# session state
if "computed_df" not in st.session_state:
    st.session_state.computed_df = None
if "playing" not in st.session_state:
    st.session_state.playing = False

# Build DF only on button click
def compute_df(hrv, rr, spo2):
    n = 15
    rows = []
    for i in range(n):
        # HRV
        h_label, d_hrv, T_hrv, E_hrv = hrv_label(hrv, i)

        # RR
        if i == 0:
            r_label, T_rr = "GREEN", 0.0
            d_rr = 0.0
        else:
            r_label, T_rr = rr_label(rr[i-1], rr[i])
            d_rr = pct_change(rr[i-1], rr[i])

        # SpO2
        if i == 0:
            s_label, T_sp = "GREEN", 0.0
            d_sp = 0.0
        else:
            next_val = spo2[i+1] if i + 1 < n else None
            s_label, T_sp = spo2_label(spo2[i-1], spo2[i], next_val=next_val)
            d_sp = spo2[i] - spo2[i-1]

        dn = dn_gatekeeper(h_label, r_label, s_label)

        dn_level = 0
        if dn == "WARNING":
            dn_level = 1
        elif dn == "RED":
            dn_level = 2

        rows.append({
            "minute": i + 1,
            "HRV": hrv[i],
            "RR": rr[i],
            "SpO2": spo2[i],
            "%dHRV": d_hrv if i > 0 else 0.0,
            "%dRR": d_rr if i > 0 else 0.0,
            "dSpO2": d_sp if i > 0 else 0.0,
            "T_hrv": T_hrv,
            "E_hrv": E_hrv,
            "T_rr": T_rr,
            "T_spo2": T_sp,
            "HRV_status": h_label,
            "RR_status": r_label,
            "SpO2_status": s_label,
            "DN_status": dn,
            "DN_level": dn_level
        })
    return pd.DataFrame(rows)

def validate_inputs(hrv, rr, spo2):
    return len(hrv) == 15 and len(rr) == 15 and len(spo2) == 15

# button actions
hrv = parse_15(hrv_txt)
rr = parse_15(rr_txt)
spo2 = parse_15(spo2_txt)

if static_btn or play_btn:
    if not validate_inputs(hrv, rr, spo2):
        with right:
            st.error("Bạn phải nhập đúng **15 số** cho mỗi dãy HRV / RR / SpO₂.")
        st.stop()
    st.session_state.computed_df = compute_df(hrv, rr, spo2)

# Render output
with right:
    st.subheader("Output")

    df = st.session_state.computed_df
    if df is None:
        st.info("Nhập 15 điểm cho mỗi chỉ số, rồi bấm **Run static** hoặc **Play**.")
        st.stop()

    # Summary minutes
    red_minutes = df.loc[df["DN_status"] == "RED", "minute"].tolist()
    warn_minutes = df.loc[df["DN_status"] == "WARNING", "minute"].tolist()

    st.write(f"**DN RED minutes:** {red_minutes if red_minutes else 'None'}")
    st.write(f"**DN WARNING minutes:** {warn_minutes if warn_minutes else 'None'}")

    ph1 = st.empty()
    ph2 = st.empty()
    ph3 = st.empty()

    def render(upto):
        ph1.plotly_chart(plot_dn(df, upto), use_container_width=True)
        ph2.plotly_chart(plot_raw(df, upto), use_container_width=True)

        with ph3.container():
            st.markdown("### Table (per minute)")
            st.dataframe(df.iloc[:upto], use_container_width=True, height=320)

    # Static
    if static_btn and not play_btn:
        render(15)

    # Play
    if play_btn:
        for upto in range(1, 16):
            render(upto)
            time.sleep(3)
