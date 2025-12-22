import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="DN Alerts – HRV / SpO₂ / RR", layout="wide")

# -----------------------------
# Helpers
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

def demo_series(signal="HRV", n=50, seed=7):
    rng = np.random.default_rng(seed)
    if signal == "HRV":
        base = 38 + rng.normal(0, 1.2, n).cumsum() * 0.05
        base = np.clip(base, 10, 160)
        base[18] += 14; base[19] -= 12
        base[34:40] -= np.linspace(0, 10, 6)
        return base.tolist()
    if signal == "SpO₂":
        base = 97 + rng.normal(0, 0.25, n)
        base[12] = 92; base[13] = 97   # drop + rebound (false alarm-like)
        base[32:50] -= np.linspace(0, 7, 18)
        return np.clip(base, 82, 100).tolist()
    if signal == "RR":
        base = 16 + rng.normal(0, 0.35, n)
        base[20:28] += np.linspace(0, 6, 8)
        base[28:35] += 4
        return np.clip(base, 8, 40).tolist()
    return rng.normal(0, 1, n).tolist()

# -----------------------------
# Config: “chuẩn cảnh báo” (gọn, ít tham số)
# - HRV, SpO2: tụt (down) là xấu
# - RR: tăng (up) là xấu
# Logic chính: rolling total % over WINDOW + ngưỡng thô (raw)
# -----------------------------
CFG = {
    "HRV": dict(
        window=10,                 # 10 phút
        warn_total=-25.0,          # tổng tụt ≤ -25% -> WARNING
        red_total=-40.0,           # tổng tụt ≤ -40% -> RED
        raw_warn=28.0,             # HRV thô thấp -> WARNING
        raw_red=24.0,              # HRV thô rất thấp -> RED
        noise_up_pct=70.0,         # step tăng quá nhanh -> noise brake
        noise_abs_jump=60.0,       # jump ms lớn -> noise brake
        direction="down_bad",
        naive_step=-20.0           # naive đỏ nếu step ≤ -20%
    ),
    "SpO₂": dict(
        window=10,
        warn_total=-3.0,           # tổng tụt % theo SpO2 là nhỏ (vd 97->94 ~ -3.1%)
        red_total=-6.0,
        raw_warn=92.0,
        raw_red=89.0,
        noise_up_pct=3.0,          # SpO2 tăng step >3% thường là artefact
        noise_abs_jump=None,
        direction="down_bad",
        naive_step=-2.0
    ),
    "RR": dict(
        window=10,
        warn_total=18.0,           # RR tăng tổng ≥ +18% -> WARNING
        red_total=30.0,            # RR tăng tổng ≥ +30% -> RED
        raw_warn=22.0,
        raw_red=28.0,
        noise_up_pct=80.0,
        noise_abs_jump=None,
        direction="up_bad",
        naive_step=20.0            # naive đỏ nếu step ≥ +20%
    )
}

# -----------------------------
# DN engine (window kinetics + raw thresholds + noise brake + V-shape)
# -----------------------------
def dn_engine(raw, signal_name):
    cfg = CFG[signal_name]
    x = np.array(raw, dtype=float)
    n = len(x)
    t = make_time_index(n)

    # step %
    pct_step = np.zeros(n, dtype=float)
    pct_step[1:] = 100.0 * (x[1:] - x[:-1]) / np.clip(x[:-1], 1e-9, None)

    # rolling total % over window (i - W)
    W = int(cfg["window"])
    total_pct = np.zeros(n, dtype=float)
    for i in range(n):
        j = i - W
        if j >= 0:
            total_pct[i] = 100.0 * (x[i] - x[j]) / np.clip(x[j], 1e-9, None)
        else:
            total_pct[i] = 0.0

    # naive (to show “báo loạn”)
    if cfg["direction"] == "down_bad":
        naive_alert = (pct_step <= cfg["naive_step"])
    else:
        naive_alert = (pct_step >= cfg["naive_step"])

    # noise brake
    noise = np.zeros(n, dtype=bool)
    noise |= (pct_step >= cfg["noise_up_pct"])
    if cfg["noise_abs_jump"] is not None:
        abs_jump = np.zeros(n, dtype=float)
        abs_jump[1:] = np.abs(x[1:] - x[:-1])
        noise |= (abs_jump >= float(cfg["noise_abs_jump"]))

    # V-shape recovery (đơn giản): 1 bước tụt mạnh + 1 bước hồi mạnh
    vshape = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if cfg["direction"] == "down_bad":
            if (pct_step[i-1] <= -20.0) and (pct_step[i] >= +15.0):
                vshape[i] = True
        else:
            if (pct_step[i-1] >= +20.0) and (pct_step[i] <= -15.0):
                vshape[i] = True

    alert = np.array(["GREEN"] * n, dtype=object)
    tag = np.array([""] * n, dtype=object)

    for i in range(n):
        if i == 0:
            alert[i] = "GREEN"; tag[i] = "baseline"; continue

        if noise[i]:
            alert[i] = "INFO"; tag[i] = "noise_brake"; continue

        if vshape[i]:
            alert[i] = "INFO"; tag[i] = "V_recovery"; continue

        # raw thresholds (fast safety)
        if cfg["direction"] == "down_bad":
            if x[i] <= cfg["raw_red"]:
                alert[i] = "RED"; tag[i] = "raw_low"; continue
            if x[i] <= cfg["raw_warn"]:
                alert[i] = "WARNING"; tag[i] = "raw_low"; continue
        else:
            if x[i] >= cfg["raw_red"]:
                alert[i] = "RED"; tag[i] = "raw_high"; continue
            if x[i] >= cfg["raw_warn"]:
                alert[i] = "WARNING"; tag[i] = "raw_high"; continue

        # kinetics (window total)
        if cfg["direction"] == "down_bad":
            if total_pct[i] <= cfg["red_total"]:
                alert[i] = "RED"; tag[i] = f"total_{W}"; continue
            if total_pct[i] <= cfg["warn_total"]:
                alert[i] = "WARNING"; tag[i] = f"total_{W}"; continue
        else:
            if total_pct[i] >= cfg["red_total"]:
                alert[i] = "RED"; tag[i] = f"total_{W}"; continue
            if total_pct[i] >= cfg["warn_total"]:
                alert[i] = "WARNING"; tag[i] = f"total_{W}"; continue

        alert[i] = "GREEN"; tag[i] = "stable"

    df = pd.DataFrame({
        "time": t,
        "raw": x,
        "pct_step": pct_step,
        "total_pct_window": total_pct,
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

def add_alert_bands(fig, seg_df, y0, y1):
    band_color = {
        "GREEN": "rgba(0, 200, 0, 0.10)",
        "INFO": "rgba(0, 160, 255, 0.12)",
        "WARNING": "rgba(255, 180, 0, 0.14)",
        "RED": "rgba(255, 0, 0, 0.18)",
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
    pad = (y_max - y_min) * 0.18 if y_max > y_min else 1.0
    y0, y1 = y_min - pad, y_max + pad

    fig = go.Figure()
    seg = segments(df)
    add_alert_bands(fig, seg, y0, y1)

    fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", name="Raw"))

    mask = df["naive_alert"].values.astype(bool)
    fig.add_trace(go.Scatter(
        x=t[mask],
        y=y[mask],
        mode="markers",
        name="Naive alert (red dots)",
        marker=dict(color="red", size=10, symbol="x")
    ))

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="time",
        yaxis_title="raw",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_yaxes(range=[y0, y1])
    return fig

# -----------------------------
# UI (2 columns, no "Parameters" block)
# -----------------------------
st.title("DN Alerts – HRV / SpO₂ / RR (50×1min)")
st.caption("Cột trái nhập số + Run. Cột phải hiển thị đồ thị (DN bands + naive red dots).")

tabs = st.tabs(["HRV", "SpO₂", "RR"])

def render_tab(signal):
    left, right = st.columns([1, 2], gap="large")

    cfg = CFG[signal]

    with left:
        st.subheader(f"{signal} input")

        if signal == "HRV":
            default_text = "45 44 42 35 36 32 30 28 27 22"
        elif signal == "SpO₂":
            default_text = "98 97 92 97 95 94 94 92 93 89"
        else:
            default_text = "16 16 17 18 20 22 24 24 23 22"

        text = st.text_area("Paste values (space-separated)", value=default_text, height=120)

        c1, c2 = st.columns(2)
        with c1:
            use_demo = st.checkbox("Use demo 50 points", value=False)
        with c2:
            seed = st.number_input("Demo seed", 0, 9999, 7, 1)

        run = st.button("Run", type="primary")

        st.markdown("**Current alert rules (fixed):**")
        if cfg["direction"] == "down_bad":
            st.write(f"- **RED** if raw ≤ {cfg['raw_red']} OR total({cfg['window']}m) ≤ {cfg['red_total']}%")
            st.write(f"- **WARNING** if raw ≤ {cfg['raw_warn']} OR total({cfg['window']}m) ≤ {cfg['warn_total']}%")
        else:
            st.write(f"- **RED** if raw ≥ {cfg['raw_red']} OR total({cfg['window']}m) ≥ {cfg['red_total']}%")
            st.write(f"- **WARNING** if raw ≥ {cfg['raw_warn']} OR total({cfg['window']}m) ≥ {cfg['warn_total']}%")
        st.write(f"- **INFO** noise_brake if step% ≥ {cfg['noise_up_pct']} (or big jump), or V-recovery")
        st.write(f"- **Naive red dots**: step% threshold = {cfg['naive_step']} ({'≤' if cfg['direction']=='down_bad' else '≥'})")

    if not run:
        with right:
            st.info("Nhập dữ liệu rồi bấm **Run**.")
        return

    # build series
    if use_demo:
        raw = demo_series(signal, 50, int(seed))
    else:
        raw = normalize_len(parse_series(text), 50)

    if len(raw) == 0:
        with right:
            st.error("Không đọc được số nào. Bạn dán lại chuỗi số (cách nhau bởi space/comma).")
        return

    df = dn_engine(raw, signal)

    # summary metrics
    counts = df["alert"].value_counts().reindex(["GREEN", "INFO", "WARNING", "RED"]).fillna(0).astype(int)
    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("GREEN", int(counts["GREEN"]))
        m2.metric("INFO", int(counts["INFO"]))
        m3.metric("WARNING", int(counts["WARNING"]))
        m4.metric("RED", int(counts["RED"]))

        st.plotly_chart(plot_raw_with_dn_and_naive(df, f"{signal}: Raw + DN bands + naive red dots"), use_container_width=True)

    # tables (below, still gọn)
    st.markdown("### Table")
    show = df.copy()
    for col in ["raw", "pct_step", "total_pct_window"]:
        show[col] = show[col].astype(float).round(4)
    st.dataframe(show, use_container_width=True, height=420)

    st.markdown("### Segments")
    st.dataframe(segments(df), use_container_width=True)

    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button(f"Download {signal} CSV", data=csv, file_name=f"DN_{signal}_50points.csv", mime="text/csv")

with tabs[0]:
    render_tab("HRV")
with tabs[1]:
    render_tab("SpO₂")
with tabs[2]:
    render_tab("RR")
