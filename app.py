# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="DN Alerts – HRV / SpO2 / RR (50×1min)", layout="wide")

# -----------------------------
# Utils
# -----------------------------
def parse_series(text: str):
    if not text or not text.strip():
        return []
    cleaned = text.replace(",", " ").replace("\n", " ").replace("\t", " ")
    parts = [p for p in cleaned.split(" ") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except:
            pass
    return out

def normalize_len(raw, n=50):
    if len(raw) == 0:
        return []
    if len(raw) < n:
        raw = raw + [raw[-1]] * (n - len(raw))
    if len(raw) > n:
        raw = raw[:n]
    return raw

def make_time_index(n=50):
    start = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=n-1)
    return [start + timedelta(minutes=i) for i in range(n)]

# -----------------------------
# SpO2 fixed physiological weight (ceiling compensation)
# -----------------------------
def w_spo2(v):
    if v >= 97: return 0.3
    if v >= 95: return 0.6
    if v >= 92: return 1.0
    return 1.7

# -----------------------------
# DN engine (single-signal)
# -----------------------------
def dn_engine(
    raw,
    signal_name="HRV",
    K=80.0,
    use_spo2_weight=False,
    K_spo2=5.0,
    warn_pct_eff=-15.0,
    red_pct_eff=-20.0,
    red_consecutive=3,
    vshape_drop=-20.0,
    vshape_recover=15.0,
    noise_up_pct=70.0,
    noise_abs_jump=None,
    naive_threshold=-20.0
):
    """
    Returns df with:
      raw, pct_step, w, pct_eff, TT, E, vE, alert, tag, naive_alert
    """
    x = np.array(raw, dtype=float)
    n = len(x)
    t = make_time_index(n)

    pct = np.zeros(n, dtype=float)
    pct[1:] = 100.0 * (x[1:] - x[:-1]) / np.clip(x[:-1], 1e-9, None)

    if use_spo2_weight:
        w = np.array([w_spo2(v) for v in x], dtype=float)
        pct_eff = pct * w
        TT = pct_eff / float(K_spo2)
    else:
        w = np.ones(n, dtype=float)
        pct_eff = pct
        TT = pct_eff / float(K)

    E = 1.0 - TT**2
    vE = np.zeros(n, dtype=float)
    vE[1:] = E[1:] - E[:-1]

    # naive alert based on raw %Δ only
    naive_alert = (pct <= naive_threshold)

    # noise brake (simple)
    noise = np.zeros(n, dtype=bool)
    noise |= (pct >= noise_up_pct)
    if noise_abs_jump is not None:
        abs_jump = np.zeros(n, dtype=float)
        abs_jump[1:] = np.abs(x[1:] - x[:-1])
        noise |= (abs_jump >= float(noise_abs_jump))

    # V-shape recovery on pct_eff
    vshape = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if (pct_eff[i-1] <= vshape_drop) and (pct_eff[i] >= vshape_recover):
            vshape[i] = True

    # sustained deterioration on pct_eff
    sustained = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < red_consecutive:
            continue
        window = pct_eff[i-red_consecutive+1:i+1]
        window_noise = noise[i-red_consecutive+1:i+1]
        if np.all(window <= red_pct_eff) and (not np.any(window_noise)):
            sustained[i] = True

    alert = np.array(["GREEN"] * n, dtype=object)
    tag = np.array([""] * n, dtype=object)

    for i in range(n):
        if i == 0:
            alert[i] = "GREEN"; tag[i] = "baseline"; continue
        if noise[i]:
            alert[i] = "INFO"; tag[i] = "noise_brake"; continue
        if sustained[i]:
            alert[i] = "RED"; tag[i] = f"sustained_{red_consecutive}"; continue
        if vshape[i]:
            alert[i] = "INFO"; tag[i] = "V_recovery"; continue
        if pct_eff[i] <= warn_pct_eff:
            alert[i] = "WARNING"; tag[i] = "early_drop"; continue
        alert[i] = "GREEN"; tag[i] = "stable"

    df = pd.DataFrame({
        "time": t,
        "raw": x,
        "pct_step": pct,
        "w": w,
        "pct_eff": pct_eff,
        "TT": TT,
        "E": E,
        "vE": vE,
        "alert": alert,
        "tag": tag,
        "naive_alert": naive_alert
    })
    df.insert(0, "signal", signal_name)
    return df

def segments(df):
    segs = []
    start = 0
    for i in range(1, len(df)):
        if df.loc[i, "alert"] != df.loc[i-1, "alert"]:
            segs.append((df.loc[start, "time"], df.loc[i-1, "time"], df.loc[start, "alert"]))
            start = i
    segs.append((df.loc[start, "time"], df.loc[len(df)-1, "time"], df.loc[start, "alert"]))
    return pd.DataFrame(segs, columns=["start", "end", "alert"])

# -----------------------------
# Plotly charts
# -----------------------------
def add_alert_bands(fig, seg_df, y0, y1):
    # NOTE: Plotly doesn't accept color names for opacity-free reliably across themes;
    # we use rgba with fixed transparency.
    band_color = {
        "GREEN": "rgba(0, 200, 0, 0.10)",
        "INFO": "rgba(0, 160, 255, 0.12)",
        "WARNING": "rgba(255, 180, 0, 0.14)",
        "RED": "rgba(255, 0, 0, 0.16)",
    }
    for _, r in seg_df.iterrows():
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=r["start"], x1=r["end"],
            y0=y0, y1=y1,
            fillcolor=band_color.get(r["alert"], "rgba(0,0,0,0.08)"),
            line_width=0,
            layer="below"
        )

def plot_raw_with_dn_and_naive(df, title):
    t = df["time"]
    y = df["raw"]

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    pad = (y_max - y_min) * 0.15 if y_max > y_min else 1.0
    y0, y1 = y_min - pad, y_max + pad

    fig = go.Figure()

    # DN alert bands
    seg = segments(df)
    add_alert_bands(fig, seg, y0, y1)

    # raw line
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", name="Raw"))

    # naive alerts: red markers on raw when naive_alert=True
    mask = df["naive_alert"].values.astype(bool)
    fig.add_trace(go.Scatter(
        x=t[mask],
        y=y[mask],
        mode="markers",
        name="Naive %Δ alert (red dots)",
        marker=dict(color="red", size=10, symbol="x")
    ))

    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="time",
        yaxis_title="raw",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_yaxes(range=[y0, y1])
    return fig

def plot_pct_compare(df, title, naive_threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["pct_step"], mode="lines+markers", name="%Δ_step (naive)"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["pct_eff"], mode="lines+markers", name="%Δ_eff (DN input)"))
    fig.add_hline(y=naive_threshold, line_dash="dash", annotation_text=f"naive threshold {naive_threshold}")
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="time",
        yaxis_title="% change",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# -----------------------------
# Demo generators (optional)
# -----------------------------
def demo_series(signal="HRV", n=50, seed=7):
    rng = np.random.default_rng(seed)
    if signal == "HRV":
        base = 35 + rng.normal(0, 1.2, n).cumsum() * 0.06
        base = np.clip(base, 12, 160)
        base[18] += 14; base[19] -= 12
        base[34:40] -= np.linspace(0, 10, 6)
        return base.tolist()
    if signal == "SpO₂":
        base = 97 + rng.normal(0, 0.2, n)
        base[12] = 92; base[13] = 97
        base[32:50] -= np.linspace(0, 7, 18)
        return np.clip(base, 82, 100).tolist()
    if signal == "RR":
        base = 16 + rng.normal(0, 0.35, n)
        base[20:28] += np.linspace(0, 6, 8)
        base[28:35] += 4
        return np.clip(base, 8, 40).tolist()
    return rng.normal(0, 1, n).tolist()

# -----------------------------
# UI
# -----------------------------
st.title("DN Alerts – HRV / SpO₂ / RR • 50 points • 1-min sampling")
st.caption("Bấm Run để tính. Đồ thị 1: Raw + DN band + chấm đỏ (naive %Δ báo loạn). Đồ thị 2: %Δ_step vs %Δ_eff.")

tabs = st.tabs(["HRV", "SpO₂", "RR"])

def tab(signal):
    st.subheader(signal)

    with st.form(key=f"form_{signal}"):
        if signal == "HRV":
            default_text = "45 44 42 35 36 32 30 28 27 22"
        elif signal == "SpO₂":
            default_text = "98 97 92 97 95 94 94 92 93 89"
        else:
            default_text = "16 16 17 18 20 22 24 24 23 22"

        text = st.text_area("Paste values (50 numbers preferred)", value=default_text, height=110)
        c1, c2, c3 = st.columns(3)
        with c1:
            use_demo = st.checkbox("Use demo 50 points", value=False)
        with c2:
            seed = st.number_input("Demo seed", 0, 9999, 7, 1)
        with c3:
            run = st.form_submit_button("Run")

    if not run:
        st.info("Dán dữ liệu rồi bấm **Run**.")
        st.stop()

    # series
    if use_demo:
        raw = demo_series(signal, 50, int(seed))
    else:
        raw = normalize_len(parse_series(text), 50)

    if len(raw) == 0:
        st.error("Không đọc được số nào. Bạn dán lại chuỗi số (cách nhau bởi space/comma).")
        st.stop()

    st.markdown("### Parameters")

    if signal == "HRV":
        a1, a2, a3, a4 = st.columns(4)
        with a1: K = st.number_input("K (HRV)", value=80.0, step=1.0)
        with a2: warn = st.number_input("WARNING if pct_eff ≤", value=-15.0, step=1.0)
        with a3: red = st.number_input("RED if pct_eff ≤", value=-20.0, step=1.0)
        with a4: consec = st.slider("RED consecutive steps", 2, 6, 3)

        b1, b2, b3, b4 = st.columns(4)
        with b1: vdrop = st.number_input("V-shape drop", value=-20.0, step=1.0)
        with b2: vrec = st.number_input("V-shape recover", value=15.0, step=1.0)
        with b3: noise_up = st.number_input("Noise-brake if pct_step ≥", value=70.0, step=5.0)
        with b4: noise_abs = st.number_input("Noise abs jump (ms) ≥", value=60.0, step=5.0)

        naive_thr = st.number_input("Naive %Δ threshold (red dots if %Δ ≤)", value=-20.0, step=1.0)

        df = dn_engine(
            raw, "HRV",
            K=K,
            use_spo2_weight=False,
            warn_pct_eff=warn, red_pct_eff=red, red_consecutive=consec,
            vshape_drop=vdrop, vshape_recover=vrec,
            noise_up_pct=noise_up, noise_abs_jump=noise_abs,
            naive_threshold=naive_thr
        )

    elif signal == "SpO₂":
        a1, a2, a3, a4 = st.columns(4)
        with a1: Kspo2 = st.number_input("Kspo2 (TT = pct_eff/Kspo2)", value=5.0, step=0.5)
        with a2: warn = st.number_input("WARNING if pct_eff ≤", value=-1.2, step=0.1)
        with a3: red = st.number_input("RED if pct_eff ≤", value=-1.8, step=0.1)
        with a4: consec = st.slider("RED consecutive steps", 2, 6, 3)

        b1, b2, b3 = st.columns(3)
        with b1: vdrop = st.number_input("V-shape drop", value=-1.0, step=0.1)
        with b2: vrec = st.number_input("V-shape recover", value=1.0, step=0.1)
        with b3: noise_up = st.number_input("Noise-brake if pct_step ≥", value=3.0, step=0.5)

        naive_thr = st.number_input("Naive %Δ threshold (red dots if %Δ ≤)", value=-2.0, step=0.1)

        df = dn_engine(
            raw, "SpO₂",
            use_spo2_weight=True, K_spo2=Kspo2,
            warn_pct_eff=warn, red_pct_eff=red, red_consecutive=consec,
            vshape_drop=vdrop, vshape_recover=vrec,
            noise_up_pct=noise_up, noise_abs_jump=None,
            naive_threshold=naive_thr
        )

    else:  # RR
        a1, a2, a3, a4 = st.columns(4)
        with a1: K = st.number_input("K (RR)", value=40.0, step=1.0)
        with a2: warn = st.number_input("WARNING if pct_eff ≤", value=-10.0, step=1.0)
        with a3: red = st.number_input("RED if pct_eff ≤", value=-15.0, step=1.0)
        with a4: consec = st.slider("RED consecutive steps", 2, 6, 3)

        b1, b2, b3 = st.columns(3)
        with b1: vdrop = st.number_input("V-shape drop", value=-10.0, step=1.0)
        with b2: vrec = st.number_input("V-shape recover", value=8.0, step=1.0)
        with b3: noise_up = st.number_input("Noise-brake if pct_step ≥", value=60.0, step=5.0)

        naive_thr = st.number_input("Naive %Δ threshold (red dots if %Δ ≤)", value=-15.0, step=1.0)

        df = dn_engine(
            raw, "RR",
            K=K,
            use_spo2_weight=False,
            warn_pct_eff=warn, red_pct_eff=red, red_consecutive=consec,
            vshape_drop=vdrop, vshape_recover=vrec,
            noise_up_pct=noise_up, noise_abs_jump=None,
            naive_threshold=naive_thr
        )

    # Summary
    counts = df["alert"].value_counts().reindex(["GREEN", "INFO", "WARNING", "RED"]).fillna(0).astype(int)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GREEN", int(counts["GREEN"]))
    c2.metric("INFO", int(counts["INFO"]))
    c3.metric("WARNING", int(counts["WARNING"]))
    c4.metric("RED", int(counts["RED"]))

    # Charts
    st.markdown("### Chart 1: Raw + DN band + Naive red dots")
    st.plotly_chart(plot_raw_with_dn_and_naive(df, f"{signal}: Raw + DN bands + naive red dots"), use_container_width=True)

    st.markdown("### Chart 2: %Δ_step vs %Δ_eff (DN input)")
    st.plotly_chart(plot_pct_compare(df, f"{signal}: %Δ compare", naive_threshold=naive_thr), use_container_width=True)

    # Tables
    st.markdown("### DN Table (paper-ready)")
    show = df.copy()
    for col in ["raw", "pct_step", "w", "pct_eff", "TT", "E", "vE"]:
        show[col] = show[col].astype(float).round(4)
    st.dataframe(show, use_container_width=True, height=520)

    st.markdown("### Alert segments (đoạn xanh/vàng/đỏ)")
    st.dataframe(segments(df), use_container_width=True)

    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"DN_{signal}_50points.csv", mime="text/csv")

with tabs[0]:
    tab("HRV")
with tabs[1]:
    tab("SpO₂")
with tabs[2]:
    tab("RR")
