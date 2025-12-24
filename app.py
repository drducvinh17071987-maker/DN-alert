import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Config (chốt theo bạn)
# =========================
@dataclass
class DNConfig:
    # Normalization constants (fixed)
    K_HRV_DROP_PCT: float = 40.0     # 40% drop ~ severe step (low false alarm friendly)
    K_RR_UP_PCT: float = 25.0        # chốt theo bạn (saved)
    K_SPO2_DROP_ABS: float = 5.0     # chốt theo bạn

    # EWMA drift
    ALPHA: float = 0.80             # prioritize low false alarms (smoother)
    PERSIST_MIN: int = 3            # need >=3 consecutive minutes

    # Drift thresholds (tune for low false alarms)
    WARN_THR: float = 0.25
    RED_THR: float = 0.35

    # SHOCK thresholds (between two points) - strict to reduce false alarms
    SHOCK_COMPONENT_THR: float = 0.85   # any single system huge step
    SHOCK_FUSION_THR: float = 0.75      # mean-active huge step

    # Window for "worst-in-window"
    WORST_WINDOW: int = 5


CFG = DNConfig()


# =========================
# Helpers
# =========================
def parse_series(text: str, n: int = 15) -> List[float]:
    """
    Accepts spaces/commas/newlines.
    """
    if text is None:
        return []
    cleaned = text.replace(",", " ").replace("\n", " ").strip()
    if not cleaned:
        return []
    parts = [p for p in cleaned.split(" ") if p.strip() != ""]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            pass
    return vals[:n]


def pct_change(prev: float, cur: float) -> float:
    if prev == 0:
        return 0.0
    return 100.0 * (cur - prev) / prev


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def mean_active(values: List[float]) -> float:
    active = [v for v in values if v > 0]
    if not active:
        return 0.0
    return float(sum(active) / len(active))


def rolling_max(arr: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        out[i] = np.max(arr[lo:i+1])
    return out


def consecutive_flags(x: np.ndarray, thr: float, k: int) -> np.ndarray:
    """
    Returns boolean array where True means "thr exceeded for >=k consecutive steps up to i".
    """
    out = np.zeros_like(x, dtype=bool)
    run = 0
    for i, v in enumerate(x):
        if v >= thr:
            run += 1
        else:
            run = 0
        out[i] = (run >= k)
    return out


# =========================
# Core DN computation
# =========================
def compute_dn(hrv: List[float], rr: List[float], spo2: List[float], cfg: DNConfig) -> pd.DataFrame:
    n = min(len(hrv), len(rr), len(spo2))
    if n < 2:
        return pd.DataFrame()

    hrv = np.array(hrv[:n], dtype=float)
    rr = np.array(rr[:n], dtype=float)
    spo2 = np.array(spo2[:n], dtype=float)

    # Per-minute step badness (between i-1 -> i), index i corresponds to "minute i+1"
    bad_hrv = np.zeros(n)
    bad_rr = np.zeros(n)
    bad_spo2 = np.zeros(n)
    s = np.zeros(n)
    shock = np.zeros(n, dtype=bool)

    for i in range(1, n):
        # HRV: only drops are bad, measured in % drop
        hrv_pct = pct_change(hrv[i-1], hrv[i])
        hrv_drop = max(0.0, -hrv_pct)
        bad_hrv[i] = clip01(hrv_drop / cfg.K_HRV_DROP_PCT)

        # RR: only rises are bad, measured in % up
        rr_pct = pct_change(rr[i-1], rr[i])
        rr_up = max(0.0, rr_pct)
        bad_rr[i] = clip01(rr_up / cfg.K_RR_UP_PCT)

        # SpO2: absolute drop is bad (more stable than %)
        spo2_drop = max(0.0, spo2[i-1] - spo2[i])
        bad_spo2[i] = clip01(spo2_drop / cfg.K_SPO2_DROP_ABS)

        s[i] = mean_active([bad_hrv[i], bad_rr[i], bad_spo2[i]])

        # SHOCK signal (strict for low false alarms)
        if (bad_hrv[i] >= cfg.SHOCK_COMPONENT_THR or
            bad_rr[i] >= cfg.SHOCK_COMPONENT_THR or
            bad_spo2[i] >= cfg.SHOCK_COMPONENT_THR or
            s[i] >= cfg.SHOCK_FUSION_THR):
            shock[i] = True

    # EWMA drift (baseline-free, smooth)
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = cfg.ALPHA * drift[i-1] + (1.0 - cfg.ALPHA) * s[i]

    # Worst-in-window on drift (shows "approaching boundary" even if last step is small)
    worst = rolling_max(drift, cfg.WORST_WINDOW)

    # Persistence flags (>=3 consecutive minutes)
    warn_persist = consecutive_flags(worst, cfg.WARN_THR, cfg.PERSIST_MIN)
    red_persist = consecutive_flags(worst, cfg.RED_THR, cfg.PERSIST_MIN)

    # Final per-minute label:
    # - RED if shock OR red_persist
    # - WARNING if warn_persist (and not RED)
    label = np.array(["STABLE"] * n, dtype=object)
    label[(warn_persist)] = "WARNING"
    label[(red_persist)] = "RED"
    label[(shock)] = "RED"  # shock overrides

    # Build table
    minutes = np.arange(1, n + 1)
    df = pd.DataFrame({
        "minute": minutes,
        "HRV": hrv,
        "RR": rr,
        "SpO2": spo2,
        "bad_hrv": bad_hrv,
        "bad_rr": bad_rr,
        "bad_spo2": bad_spo2,
        "s_mean_active": s,
        "drift_ewma": drift,
        "worst_in_window": worst,
        "shock": shock,
        "label": label
    })
    return df


def minutes_of(df: pd.DataFrame, target: str) -> List[int]:
    if df.empty:
        return []
    return df.loc[df["label"] == target, "minute"].astype(int).tolist()


# =========================
# Plot
# =========================
def plot_dn(df: pd.DataFrame, cfg: DNConfig, upto: Optional[int] = None):
    if df.empty:
        return None

    d = df.copy()
    if upto is not None:
        d = d.iloc[:upto].copy()

    x = d["minute"].to_numpy()
    y = d["worst_in_window"].to_numpy()

    fig = plt.figure(figsize=(7.2, 3.8))
    plt.plot(x, y, marker="o", linewidth=1.8)

    # Mark RED/WARNING points
    for _, row in d.iterrows():
        if row["label"] == "RED":
            plt.scatter(row["minute"], row["worst_in_window"], s=50)
        elif row["label"] == "WARNING":
            plt.scatter(row["minute"], row["worst_in_window"], s=35)

    plt.axhline(cfg.WARN_THR, linestyle="--", linewidth=1.2)
    plt.axhline(cfg.RED_THR, linestyle="--", linewidth=1.2)

    plt.title(f"DN dynamic — Worst-in-window (W={cfg.WORST_WINDOW} min)  |  EWMA α={cfg.ALPHA}")
    plt.xlabel("Time (minute index)")
    plt.ylabel("DN (0–1)")
    plt.xticks(np.arange(1, int(d["minute"].max()) + 1, 1))
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="DN alert demo", layout="wide")
st.title("DN alert demo")
st.caption("DN_dynamic (baseline-free): EWMA drift on mean-active badness + SHOCK step alert (low false alarms).")

# Session state for play
if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_idx" not in st.session_state:
    st.session_state.play_idx = 2  # start showing from minute 2
if "last_df" not in st.session_state:
    st.session_state.last_df = pd.DataFrame()

left, right = st.columns([1, 1])

with left:
    st.subheader("Input (15 points each)")

    default_hrv = "45 44 42 35 36 32 30 28 27 22 24 23 21 20 19"
    default_rr = "14 14 15 15 16 16 17 18 19 20 21 22 23 24 25"
    default_spo2 = "98 97 92 97 95 94 94 92 93 89 90 91 90 89 89"

    hrv_txt = st.text_area("HRV (ms)", value=default_hrv, height=70)
    rr_txt = st.text_area("RR (breaths/min)", value=default_rr, height=70)
    spo2_txt = st.text_area("SpO₂ (%)", value=default_spo2, height=70)

    # Keep it simple: only window selector (you asked W=3/5; default 5 for safety)
    cfg_window = st.selectbox("Worst-in-window (minutes)", options=[3, 5], index=1)
    CFG.WORST_WINDOW = int(cfg_window)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        run_static = st.button("Run static", use_container_width=True)
    with c2:
        play_btn = st.button("Play (3s/step)", use_container_width=True)
    with c3:
        stop_btn = st.button("Stop", use_container_width=True)

# Controls
if stop_btn:
    st.session_state.playing = False
    st.session_state.play_idx = 2

if play_btn:
    st.session_state.playing = True
    st.session_state.play_idx = 2

# Parse + compute
hrv = parse_series(hrv_txt, n=15)
rr = parse_series(rr_txt, n=15)
spo2 = parse_series(spo2_txt, n=15)

df = pd.DataFrame()
if len(hrv) == 15 and len(rr) == 15 and len(spo2) == 15:
    df = compute_dn(hrv, rr, spo2, CFG)
    st.session_state.last_df = df
else:
    # Use last df if user is editing during play
    if not st.session_state.last_df.empty:
        df = st.session_state.last_df.copy()

with right:
    if df.empty:
        st.warning("Please input exactly 15 numbers for each series (HRV, RR, SpO₂).")
    else:
        # Determine which part to show
        if st.session_state.playing:
            upto = st.session_state.play_idx
        else:
            upto = None

        # Summary: minutes list (per-minute, not whole-chain)
        red_minutes = minutes_of(df.iloc[:upto] if upto else df, "RED")
        warn_minutes = minutes_of(df.iloc[:upto] if upto else df, "WARNING")

        # Status banner (based on latest shown minute)
        shown = df.iloc[:upto] if upto else df
        latest_label = shown.iloc[-1]["label"]
        if latest_label == "RED":
            st.error(f"RED (minute {int(shown.iloc[-1]['minute'])})")
        elif latest_label == "WARNING":
            st.warning(f"WARNING (minute {int(shown.iloc[-1]['minute'])})")
        else:
            st.success(f"STABLE (minute {int(shown.iloc[-1]['minute'])})")

        # Explicit per-minute conclusion
        st.write(f"**RED minutes:** {', '.join(map(str, red_minutes)) if red_minutes else 'None'}")
        st.write(f"**WARNING minutes:** {', '.join(map(str, warn_minutes)) if warn_minutes else 'None'}")

        fig = plot_dn(df, CFG, upto=upto)
        st.pyplot(fig, clear_figure=True)

        with st.expander("Details (per minute: badness → s → drift → worst → label)"):
            cols = ["minute", "bad_hrv", "bad_rr", "bad_spo2", "s_mean_active",
                    "drift_ewma", "worst_in_window", "shock", "label"]
            st.dataframe(shown[cols].style.format({
                "bad_hrv": "{:.2f}", "bad_rr": "{:.2f}", "bad_spo2": "{:.2f}",
                "s_mean_active": "{:.2f}", "drift_ewma": "{:.2f}", "worst_in_window": "{:.2f}"
            }), use_container_width=True)

# Play loop
if st.session_state.playing and not df.empty:
    # Advance 1 step each 3 seconds until minute 15
    time.sleep(3)
    st.session_state.play_idx += 1
    if st.session_state.play_idx > 15:
        st.session_state.playing = False
        st.session_state.play_idx = 2
    st.rerun()
