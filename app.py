# app.py
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Parsing helpers
# -----------------------------
def parse_series(text: str, n: int = 15) -> np.ndarray:
    """
    Accepts space/comma/newline separated numbers.
    """
    if text is None:
        return np.array([])
    s = text.replace(",", " ").replace("\n", " ").strip()
    if not s:
        return np.array([])
    parts = [p for p in s.split(" ") if p.strip() != ""]
    arr = np.array([float(p) for p in parts], dtype=float)
    if len(arr) != n:
        raise ValueError(f"Need exactly {n} values, got {len(arr)}")
    return arr

def pct_change(x: np.ndarray) -> np.ndarray:
    """% change between consecutive points. First point = 0."""
    out = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        prev = x[i-1]
        if prev == 0:
            out[i] = 0.0
        else:
            out[i] = 100.0 * (x[i] - prev) / prev
    return out

def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        j0 = max(0, i - w + 1)
        out[i] = float(np.mean(x[j0:i+1]))
    return out

# -----------------------------
# DN core (SHOCK + DRIFT)
# -----------------------------
def compute_channel_spo2(spo2: np.ndarray, k_spo2: float = 5.0, w: int = 5):
    """
    TT_shock = ΔSpO2 / K  (unit: %point per minute)
    TT_drift = (SpO2 - rolling_mean(SpO2,w)) / K
    Use abs(TT) for thresholding; keep sign for plots if needed.
    V-shape: drop then recover quickly => INFO (false-alarm filter)
    """
    n = len(spo2)
    delta = np.zeros(n)
    delta[1:] = spo2[1:] - spo2[:-1]

    tt_shock = delta / k_spo2

    rm = rolling_mean(spo2, w)
    tt_drift = (spo2 - rm) / k_spo2

    # V-shape detector (simple & conservative):
    # A sharp drop then sharp rebound within 2 steps.
    vshape = np.zeros(n, dtype=bool)
    for i in range(2, n):
        d1 = spo2[i-1] - spo2[i-2]
        d2 = spo2[i]   - spo2[i-1]
        # drop then rebound, and rebound cancels most of drop
        if d1 <= -2 and d2 >= 2 and abs((d1 + d2)) <= 1:
            vshape[i] = True

    # thresholds on |T|
    abs_shock = np.abs(tt_shock)
    abs_drift = np.abs(tt_drift)
    abs_T = np.maximum(abs_shock, abs_drift)  # combine shock+drift

    label = np.array(["GREEN"] * n, dtype=object)
    label[abs_T >= 0.3] = "WARNING"
    label[abs_T >= 0.6] = "RED"

    # V-shape overrides to INFO (filter), but only if it was WARNING/RED
    label[vshape & (label != "GREEN")] = "INFO"

    # E / vT / vE (Lorentz form) using combined T with sign from shock (fallback drift)
    # NOTE: we only need magnitude for alerting; for E we keep a signed T (choose bigger contributor)
    t_signed = np.where(np.abs(tt_shock) >= np.abs(tt_drift), tt_shock, tt_drift)
    e = 1.0 - (t_signed ** 2)
    vT = np.zeros(n)
    vE = np.zeros(n)
    vT[1:] = t_signed[1:] - t_signed[:-1]
    vE[1:] = e[1:] - e[:-1]

    return {
        "TT_shock": tt_shock, "TT_drift": tt_drift,
        "T": t_signed, "E": e, "vT": vT, "vE": vE,
        "label": label, "vshape": vshape
    }

def compute_channel_rr(rr: np.ndarray, k_rr: float = 25.0, w: int = 5):
    """
    RR uses % change per minute:
      TT_shock = (%ΔRR) / K_rr
      TT_drift = (%Δ vs rolling mean) / K_rr
    Threshold suggestion (per your note):
      |T|~0.5 ~ WARNING, |T|>=1 ~ RED
    We'll use:
      WARNING if |T|>=0.5
      RED if |T|>=1.0
    """
    n = len(rr)
    pct = pct_change(rr)  # %Δ between points
    tt_shock = pct / k_rr

    rm = rolling_mean(rr, w)
    # % difference to rolling mean:
    pct_rm = np.zeros(n)
    for i in range(n):
        base = rm[i]
        if base == 0:
            pct_rm[i] = 0.0
        else:
            pct_rm[i] = 100.0 * (rr[i] - base) / base
    tt_drift = pct_rm / k_rr

    abs_T = np.maximum(np.abs(tt_shock), np.abs(tt_drift))

    label = np.array(["GREEN"] * n, dtype=object)
    label[abs_T >= 0.5] = "WARNING"
    label[abs_T >= 1.0] = "RED"

    # V-shape filter for RR (rare, conservative): spike up then down quickly
    vshape = np.zeros(n, dtype=bool)
    for i in range(2, n):
        d1 = rr[i-1] - rr[i-2]
        d2 = rr[i]   - rr[i-1]
        if d1 >= 3 and d2 <= -3 and abs(d1 + d2) <= 1:
            vshape[i] = True
    label[vshape & (label != "GREEN")] = "INFO"

    t_signed = np.where(np.abs(tt_shock) >= np.abs(tt_drift), tt_shock, tt_drift)
    e = 1.0 - (t_signed ** 2)
    vT = np.zeros(n)
    vE = np.zeros(n)
    vT[1:] = t_signed[1:] - t_signed[:-1]
    vE[1:] = e[1:] - e[:-1]

    return {
        "TT_shock": tt_shock, "TT_drift": tt_drift,
        "T": t_signed, "E": e, "vT": vT, "vE": vE,
        "label": label, "vshape": vshape
    }

def compute_channel_hrv_shape(hrv: np.ndarray, k_hrv: float = 80.0, w: int = 5):
    """
    HRV: baseline-free shape.
    We'll implement:
      - TT_shock = (%ΔHRV)/K
      - TT_drift = (%Δ vs rolling mean)/K
      - V-shape recovery => INFO (filter)
    We DO NOT use absolute HRV level or personal baseline.
    Thresholds: keep conservative to reduce false alarms:
      WARNING if |T|>=0.35
      RED if |T|>=0.60
    (You can tune later; logic structure is what's important.)
    """
    n = len(hrv)
    pct = pct_change(hrv)
    tt_shock = pct / k_hrv

    rm = rolling_mean(hrv, w)
    pct_rm = np.zeros(n)
    for i in range(n):
        base = rm[i]
        if base == 0:
            pct_rm[i] = 0.0
        else:
            pct_rm[i] = 100.0 * (hrv[i] - base) / base
    tt_drift = pct_rm / k_hrv

    # V-shape recovery: drop then rebound within 2 steps
    vshape = np.zeros(n, dtype=bool)
    for i in range(2, n):
        d1 = hrv[i-1] - hrv[i-2]
        d2 = hrv[i]   - hrv[i-1]
        # relative to previous step size (avoid noise)
        if d1 <= -5 and d2 >= 5 and abs(d1 + d2) <= 2:
            vshape[i] = True

    abs_T = np.maximum(np.abs(tt_shock), np.abs(tt_drift))

    label = np.array(["GREEN"] * n, dtype=object)
    label[abs_T >= 0.35] = "WARNING"
    label[abs_T >= 0.60] = "RED"
    label[vshape & (label != "GREEN")] = "INFO"

    t_signed = np.where(np.abs(tt_shock) >= np.abs(tt_drift), tt_shock, tt_drift)
    e = 1.0 - (t_signed ** 2)
    vT = np.zeros(n)
    vE = np.zeros(n)
    vT[1:] = t_signed[1:] - t_signed[:-1]
    vE[1:] = e[1:] - e[:-1]

    return {
        "TT_shock": tt_shock, "TT_drift": tt_drift,
        "T": t_signed, "E": e, "vT": vT, "vE": vE,
        "label": label, "vshape": vshape
    }

def fuse_gatekeeper(lbl_hrv, lbl_rr, lbl_spo2):
    """
    DN RED if >=2 systems are RED (INFO does not count as bad)
    DN WARNING if >=2 systems are WARNING/RED
    else GREEN
    Output per-minute DN label.
    """
    n = len(lbl_hrv)
    dn = np.array(["GREEN"] * n, dtype=object)

    def is_red(x): return x == "RED"
    def is_warn_or_red(x): return x in ("WARNING", "RED")

    for i in range(n):
        reds = sum([is_red(lbl_hrv[i]), is_red(lbl_rr[i]), is_red(lbl_spo2[i])])
        warns = sum([is_warn_or_red(lbl_hrv[i]), is_warn_or_red(lbl_rr[i]), is_warn_or_red(lbl_spo2[i])])

        if reds >= 2:
            dn[i] = "RED"
        elif warns >= 2:
            dn[i] = "WARNING"
        else:
            dn[i] = "GREEN"
    return dn

# -----------------------------
# Plotting
# -----------------------------
def label_to_color(lbl: str) -> str:
    if lbl == "RED":
        return "red"
    if lbl == "WARNING":
        return "gold"
    if lbl == "INFO":
        return "deepskyblue"
    return "green"

def plot_dn(mins, dn_label, title="DN alert demo"):
    y = np.where(dn_label == "GREEN", 0, np.where(dn_label == "WARNING", 1, 2))
    fig, ax = plt.subplots()
    ax.plot(mins, y, linewidth=2)

    # scatter with true colors
    for i in range(len(mins)):
        ax.scatter(mins[i], y[i], s=60, color=label_to_color(dn_label[i]))

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["GREEN", "WARNING", "RED"])
    ax.set_xticks(mins)
    ax.set_xlabel("Minute")
    ax.set_ylabel("DN state")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_channel_T(mins, ch, name):
    T = ch["T"]
    fig, ax = plt.subplots()
    ax.plot(mins, T, linewidth=2)
    for i in range(len(mins)):
        ax.scatter(mins[i], T[i], s=35, color=label_to_color(ch["label"][i]))
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(mins)
    ax.set_xlabel("Minute")
    ax.set_ylabel("T (signed)")
    ax.set_title(f"{name}: T (shock/drift) with labels")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="DN alert demo", layout="wide")
st.title("DN alert demo")
st.caption("DN_dynamic (baseline-free): SHOCK + DRIFT + V-shape filtering, then Gatekeeper (≥2 systems).")

N = 15

if "play" not in st.session_state:
    st.session_state.play = False
if "play_i" not in st.session_state:
    st.session_state.play_i = 0
if "last_result" not in st.session_state:
    st.session_state.last_result = None

colL, colR = st.columns([1, 1])

with colL:
    st.subheader("Input (15 points each)")
    hrv_txt = st.text_area("HRV (ms)", value="40 40 41 40 39 39 38 38 37 37 36 36 35 35 35", height=90)
    rr_txt  = st.text_area("RR (breaths/min)", value="15 16 17 18 19 20 22 24 26 28 30 30 31 32 32", height=90)
    spo2_txt= st.text_area("SpO₂ (%)", value="98 97 96 95 94 93 92 91 90 90 89 89 88 88 88", height=90)

    with st.expander("Settings (hidden by default)"):
        w = st.slider("Rolling window W (minutes)", min_value=3, max_value=7, value=5, step=1)
        k_spo2 = st.number_input("K_spo2 (Δ%point per min)", value=5.0, step=0.5)
        k_rr   = st.number_input("K_rr (%Δ per min)", value=25.0, step=1.0)
        k_hrv  = st.number_input("K_hrv (%Δ per min)", value=80.0, step=5.0)

    c1, c2, c3 = st.columns([1,1,1])
    run_static = c1.button("Run static")
    play_btn   = c2.button("Play (3s/step)")
    stop_btn   = c3.button("Stop")

    st.write("Note: App updates only when you press a button (no auto-run on edit).")

def compute_all():
    hrv = parse_series(hrv_txt, N)
    rr  = parse_series(rr_txt, N)
    spo2= parse_series(spo2_txt, N)

    ch_hrv  = compute_channel_hrv_shape(hrv, k_hrv=k_hrv, w=w)
    ch_rr   = compute_channel_rr(rr, k_rr=k_rr, w=w)
    ch_spo2 = compute_channel_spo2(spo2, k_spo2=k_spo2, w=w)

    dn = fuse_gatekeeper(ch_hrv["label"], ch_rr["label"], ch_spo2["label"])

    mins = np.arange(1, N+1)

    df = pd.DataFrame({
        "minute": mins,
        "HRV": hrv, "RR": rr, "SpO2": spo2,
        "HRV_label": ch_hrv["label"],
        "RR_label": ch_rr["label"],
        "SpO2_label": ch_spo2["label"],
        "DN_label": dn,
        "HRV_T": ch_hrv["T"], "HRV_E": ch_hrv["E"], "HRV_vT": ch_hrv["vT"], "HRV_vE": ch_hrv["vE"],
        "RR_T": ch_rr["T"], "RR_E": ch_rr["E"], "RR_vT": ch_rr["vT"], "RR_vE": ch_rr["vE"],
        "SpO2_T": ch_spo2["T"], "SpO2_E": ch_spo2["E"], "SpO2_vT": ch_spo2["vT"], "SpO2_vE": ch_spo2["vE"],
    })

    return mins, ch_hrv, ch_rr, ch_spo2, dn, df

# Handle buttons
if stop_btn:
    st.session_state.play = False
    st.session_state.play_i = 0

if run_static:
    try:
        st.session_state.last_result = compute_all()
    except Exception as e:
        st.session_state.last_result = None
        with colR:
            st.error(str(e))

if play_btn:
    try:
        st.session_state.last_result = compute_all()
        st.session_state.play = True
        st.session_state.play_i = 1
    except Exception as e:
        st.session_state.last_result = None
        st.session_state.play = False
        with colR:
            st.error(str(e))

with colR:
    st.subheader("Output")

    if st.session_state.last_result is None:
        st.info("Press **Run static** or **Play** to compute.")
    else:
        mins, ch_hrv, ch_rr, ch_spo2, dn, df = st.session_state.last_result

        if st.session_state.play:
            i = st.session_state.play_i
            if i > N:
                st.session_state.play = False
                i = N

            # show partial up to i
            mins_p = mins[:i]
            dn_p = dn[:i]

            # summary at current minute
            cur = i - 1
            st.markdown(f"**Current minute:** {i}")
            st.markdown(
                f"- HRV: **{ch_hrv['label'][cur]}** | RR: **{ch_rr['label'][cur]}** | SpO₂: **{ch_spo2['label'][cur]}**"
            )
            st.markdown(f"- DN: **{dn[cur]}**")

            plot_dn(mins_p, dn_p, title="DN alert demo (PLAY)")
            # optional channel plots in play mode (comment out if you want simpler)
            # plot_channel_T(mins_p, {k: v[:i] if isinstance(v, np.ndarray) else v for k,v in ch_spo2.items()}, "SpO₂")

            with st.expander("Details (per-minute table)"):
                st.dataframe(df.iloc[:i], use_container_width=True)

            # advance
            st.session_state.play_i = i + 1
            time.sleep(3)
            st.rerun()
        else:
            # Static view
            red_mins = [int(m) for m in mins[dn == "RED"]]
            warn_mins = [int(m) for m in mins[dn == "WARNING"]]

            st.write(f"DN RED minutes: {red_mins if red_mins else 'None'}")
            st.write(f"DN WARNING minutes: {warn_mins if warn_mins else 'None'}")

            plot_dn(mins, dn, title=f"DN alert demo (STATIC) | W={w}")
            st.caption("Dots are colored by DN label per minute (GREEN/WARNING/RED).")

            with st.expander("Channel T plots (optional)"):
                plot_channel_T(mins, ch_hrv, "HRV")
                plot_channel_T(mins, ch_rr, "RR")
                plot_channel_T(mins, ch_spo2, "SpO₂")

            with st.expander("Details (per-minute table: TT/E/vT/vE + labels)"):
                st.dataframe(df, use_container_width=True)
