import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# DN alert demo (HRV · RR · SpO2)
# Baseline-free DN_dynamic:
# - Step %Δ per minute
# - Normalize by K constants (HRV=80, RR=25, SpO2=5)
# - Convert to "badness" B in [0,1] (direction-aware)
# - DN_step = mean-active of (B^2) across active systems
# - DN_worst = rolling max(DN_step, window=W)
# - Timeline labels per minute based on DN_worst
# =========================

st.set_page_config(page_title="DN alert demo", layout="wide")

# --------- constants (chốt theo bạn) ----------
K_HRV = 80.0
K_RR = 25.0
K_SPO2 = 5.0

# Thresholds for DN_worst (you can tune)
TH_WARNING = 0.30
TH_RED = 0.55

# For "urgent step" (between 2 points) - optional
URGENT_RED_STEP = 0.70  # if any single-system B^2 >= this -> step RED instantly

# Simple SpO2 sensor-noise filter (optional)
# If SpO2 drops sharply but HRV/RR do not worsen, and next minute rebounds, treat as noise.
SPO2_NOISE_DROP_B2 = 0.60   # (B^2) big drop
SPO2_NOISE_REBOUND_PCT = 2.0  # rebound >= +2% next step
OTHER_SYSTEMS_STABLE_B2 = 0.05  # HRV/RR stable


# =========================
# Helpers
# =========================
def parse_series(text: str, n_expected: int = 15):
    """
    Parse whitespace/comma-separated numbers.
    Returns list[float] length == n_expected or raises ValueError.
    """
    if text is None:
        raise ValueError("Empty input.")
    raw = text.replace(",", " ").split()
    vals = []
    for x in raw:
        try:
            vals.append(float(x))
        except:
            raise ValueError(f"Cannot parse value: {x}")
    if len(vals) != n_expected:
        raise ValueError(f"Need exactly {n_expected} values, got {len(vals)}.")
    return vals


def pct_change(x):
    """%Δ per step: 100*(x[i]-x[i-1])/x[i-1], pct[0]=0."""
    x = np.asarray(x, dtype=float)
    pct = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        denom = x[i - 1]
        pct[i] = 0.0 if denom == 0 else 100.0 * (x[i] - x[i - 1]) / denom
    return pct


def badness_from_pct(pct, mode: str, K: float):
    """
    Convert pct-change -> badness B in [0,1] direction-aware.

    mode:
      - 'drop_bad' (HRV, SpO2): negative pct is deterioration
      - 'rise_bad' (RR): positive pct is deterioration
    """
    pct = np.asarray(pct, dtype=float)
    if mode == "drop_bad":
        det = np.maximum(0.0, -pct)  # only drops are bad
    elif mode == "rise_bad":
        det = np.maximum(0.0, pct)   # only rises are bad
    else:
        raise ValueError("Unknown mode")

    T = det / float(K)  # normalized
    T = np.clip(T, 0.0, 1.0)
    E = 1.0 - T**2       # Lorentz form
    B2 = 1.0 - E         # = T^2 (severity)
    return T, E, B2


def rolling_max(a, w):
    """Rolling max with window w, aligned to current index (includes current)."""
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    for i in range(len(a)):
        j0 = max(0, i - w + 1)
        out[i] = np.max(a[j0:i+1])
    return out


def fmt_minutes(xs):
    if not xs:
        return "—"
    xs = sorted(set(int(x) for x in xs))
    ranges = []
    s = e = xs[0]
    for x in xs[1:]:
        if x == e + 1:
            e = x
        else:
            ranges.append((s, e))
            s = e = x
    ranges.append((s, e))
    out = []
    for s, e in ranges:
        out.append(str(s) if s == e else f"{s}–{e}")
    return ", ".join(out)


def compute_dn(hrv, rr, spo2, window_w: int):
    n = len(hrv)
    minutes = np.arange(1, n + 1)

    # pct-changes
    pct_hrv = pct_change(hrv)
    pct_rr = pct_change(rr)
    pct_spo2 = pct_change(spo2)

    # convert to (T,E,B2)
    T_hrv, E_hrv, B2_hrv = badness_from_pct(pct_hrv, "drop_bad", K_HRV)
    T_rr,  E_rr,  B2_rr  = badness_from_pct(pct_rr,  "rise_bad", K_RR)
    T_sp,  E_sp,  B2_sp  = badness_from_pct(pct_spo2, "drop_bad", K_SPO2)

    # ---- optional SpO2 noise filter (very simple, conservative) ----
    # If at minute i: SpO2 severe drop, HRV/RR stable, and minute i+1 rebounds -> treat minute i as noise.
    spo2_noise = np.zeros(n, dtype=bool)
    for i in range(1, n-1):
        if (B2_sp[i] >= SPO2_NOISE_DROP_B2) and (B2_hrv[i] <= OTHER_SYSTEMS_STABLE_B2) and (B2_rr[i] <= OTHER_SYSTEMS_STABLE_B2):
            if pct_spo2[i+1] >= SPO2_NOISE_REBOUND_PCT:
                spo2_noise[i] = True
                B2_sp[i] = 0.0
                T_sp[i] = 0.0
                E_sp[i] = 1.0

    # mean-active fusion across systems
    B2_stack = np.vstack([B2_hrv, B2_rr, B2_sp]).T  # shape (n,3)
    active = B2_stack > 0.0
    dn_step = np.zeros(n, dtype=float)
    active_count = np.sum(active, axis=1)
    for i in range(n):
        if active_count[i] == 0:
            dn_step[i] = 0.0
        else:
            dn_step[i] = float(np.mean(B2_stack[i, active[i]]))

    # urgent step flag
    urgent_red = np.any(B2_stack >= URGENT_RED_STEP, axis=1)

    # worst in window
    dn_worst = rolling_max(dn_step, window_w)

    # labels per minute (timeline)
    label = np.array(["GREEN"] * n, dtype=object)
    label[dn_worst >= TH_WARNING] = "WARNING"
    label[dn_worst >= TH_RED] = "RED"
    # urgent overrides
    label[urgent_red] = "RED"

    # vT / vE (per-system) and vDN
    vT_hrv = np.diff(T_hrv, prepend=T_hrv[0])
    vE_hrv = np.diff(E_hrv, prepend=E_hrv[0])
    vT_rr  = np.diff(T_rr,  prepend=T_rr[0])
    vE_rr  = np.diff(E_rr,  prepend=E_rr[0])
    vT_sp  = np.diff(T_sp,  prepend=T_sp[0])
    vE_sp  = np.diff(E_sp,  prepend=E_sp[0])
    vDN    = np.diff(dn_worst, prepend=dn_worst[0])

    df = pd.DataFrame({
        "min": minutes,
        "HRV": hrv,
        "RR": rr,
        "SpO2": spo2,

        "%dHRV": pct_hrv,
        "%dRR": pct_rr,
        "%dSpO2": pct_spo2,

        "T_hrv": T_hrv, "E_hrv": E_hrv, "B2_hrv": B2_hrv, "vT_hrv": vT_hrv, "vE_hrv": vE_hrv,
        "T_rr":  T_rr,  "E_rr":  E_rr,  "B2_rr":  B2_rr,  "vT_rr":  vT_rr,  "vE_rr":  vE_rr,
        "T_spo2": T_sp, "E_spo2": E_sp, "B2_spo2": B2_sp, "vT_spo2": vT_sp, "vE_spo2": vE_sp,

        "DN_step": dn_step,
        "DN_worst": dn_worst,
        "vDN": vDN,
        "urgent_red": urgent_red,
        "spo2_noise_filtered": spo2_noise,
        "label": label
    })
    return df


def plot_dn(df, title="DN dynamic (Worst-in-window)", show_upto=None):
    if show_upto is None:
        show_upto = len(df)
    d = df.iloc[:show_upto].copy()

    x = d["min"].values
    y = d["DN_worst"].values
    lab = d["label"].values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, y, linewidth=2)

    # colored points
    for xi, yi, li in zip(x, y, lab):
        if li == "RED":
            ax.scatter([xi], [yi], s=45, marker="o")
        elif li == "WARNING":
            ax.scatter([xi], [yi], s=45, marker="o")
        else:
            ax.scatter([xi], [yi], s=35, marker="o")

    # thresholds
    ax.axhline(TH_WARNING, linestyle="--", linewidth=1)
    ax.axhline(TH_RED, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Time (minute index)")
    ax.set_ylabel("DN_worst (rolling max)")
    ax.set_xticks(list(range(1, int(df["min"].max()) + 1)))
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25)
    return fig


# =========================
# UI
# =========================
st.title("DN alert demo")
st.caption("DN_dynamic (baseline-free) • HRV + RR + SpO₂ • Worst-in-window + timeline RED/WARNING per minute")

if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_idx" not in st.session_state:
    st.session_state.play_idx = 15

colL, colR = st.columns([1.05, 1.2], gap="large")

with colL:
    st.subheader("Input (15 points each)")
    default_hrv = "45 44 42 35 36 32 30 28 27 22 24 23 21 20 19"
    default_rr  = "14 14 15 15 16 16 17 18 19 20 21 22 23 24 25"
    default_sp  = "98 97 92 97 95 94 94 92 93 89 90 91 90 89 89"

    hrv_txt = st.text_area("HRV (ms)", value=default_hrv, height=75)
    rr_txt  = st.text_area("RR (breaths/min)", value=default_rr, height=75)
    sp_txt  = st.text_area("SpO₂ (%)", value=default_sp, height=75)

    window_w = st.selectbox("Worst-in-window (minutes)", options=[3, 5], index=1)

    c1, c2, c3 = st.columns([1, 1, 1])
    run_static = c1.button("Run static", use_container_width=True)
    play_btn   = c2.button("Play (3s/step)", use_container_width=True)
    stop_btn   = c3.button("Stop", use_container_width=True)

    st.markdown("---")
    with st.expander("What this app computes (quick)", expanded=False):
        st.markdown(
            "- Step %Δ per minute for each series.\n"
            "- Direction-aware deterioration:\n"
            "  - HRV drop is bad, RR rise is bad, SpO₂ drop is bad.\n"
            "- Normalize by K: HRV=80, RR=25, SpO₂=5.\n"
            "- Per-system: T = det/K, E = 1 - T², severity = B² = 1 - E.\n"
            "- Fuse (mean-active): DN_step = mean(B² of active systems).\n"
            "- Worst-in-window: DN_worst = rolling max(DN_step, W).\n"
            "- Label per minute from DN_worst (and urgent step override)."
        )

with colR:
    status_box = st.empty()
    chart_box = st.empty()
    details_box = st.empty()

def run_and_render(show_upto=None):
    try:
        hrv = parse_series(hrv_txt, 15)
        rr  = parse_series(rr_txt, 15)
        sp  = parse_series(sp_txt, 15)
    except Exception as e:
        status_box.error(str(e))
        return

    df = compute_dn(hrv, rr, sp, window_w=int(window_w))

    # Timeline conclusion: minute-by-minute
    red_minutes = df.loc[df["label"] == "RED", "min"].astype(int).tolist()
    warn_minutes = df.loc[df["label"] == "WARNING", "min"].astype(int).tolist()

    # Show status (based on latest visible minute)
    if show_upto is None:
        show_upto = len(df)
    last_label = df.iloc[show_upto - 1]["label"]

    if last_label == "RED":
        status_box.error(
            f"🔴 RED at minute {int(df.iloc[show_upto-1]['min'])}  •  Worst-in-window={df.iloc[show_upto-1]['DN_worst']:.3f}\n\n"
            f"**RED minutes:** {fmt_minutes(red_minutes)}\n\n"
            f"**WARNING minutes:** {fmt_minutes(warn_minutes)}"
        )
    elif last_label == "WARNING":
        status_box.warning(
            f"🟠 WARNING at minute {int(df.iloc[show_upto-1]['min'])}  •  Worst-in-window={df.iloc[show_upto-1]['DN_worst']:.3f}\n\n"
            f"**RED minutes:** {fmt_minutes(red_minutes)}\n\n"
            f"**WARNING minutes:** {fmt_minutes(warn_minutes)}"
        )
    else:
        status_box.success(
            f"🟢 STABLE at minute {int(df.iloc[show_upto-1]['min'])}  •  Worst-in-window={df.iloc[show_upto-1]['DN_worst']:.3f}\n\n"
            f"**RED minutes:** {fmt_minutes(red_minutes)}\n\n"
            f"**WARNING minutes:** {fmt_minutes(warn_minutes)}"
        )

    fig = plot_dn(df, title=f"DN dynamic — Worst-in-window (W={window_w} min)", show_upto=show_upto)
    chart_box.pyplot(fig, clear_figure=True)

    with details_box.container():
        with st.expander("Details (per minute: %Δ, T/E, vT/vE, DN_step/DN_worst, flags)", expanded=False):
            show_cols = [
                "min",
                "HRV", "RR", "SpO2",
                "%dHRV", "%dRR", "%dSpO2",
                "T_hrv", "E_hrv", "vT_hrv", "vE_hrv",
                "T_rr", "E_rr", "vT_rr", "vE_rr",
                "T_spo2", "E_spo2", "vT_spo2", "vE_spo2",
                "DN_step", "DN_worst", "vDN",
                "urgent_red", "spo2_noise_filtered",
                "label",
            ]
            st.dataframe(df[show_cols].round(4), use_container_width=True)

# Button logic
if stop_btn:
    st.session_state.playing = False

if run_static:
    st.session_state.playing = False
    st.session_state.play_idx = 15
    run_and_render(show_upto=15)

if play_btn:
    st.session_state.playing = True
    st.session_state.play_idx = 1

# Play loop (simple)
if st.session_state.playing:
    idx = st.session_state.play_idx
    idx = max(1, min(15, idx))
    run_and_render(show_upto=idx)
    if idx >= 15:
        st.session_state.playing = False
    else:
        time.sleep(3)
        st.session_state.play_idx = idx + 1
        st.rerun()
else:
    # default render (full)
    run_and_render(show_upto=15)
