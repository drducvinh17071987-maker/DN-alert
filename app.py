import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------
def parse_series(text, n=15):
    vals = []
    for tok in text.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except:
            pass
    if len(vals) < n:
        vals = vals + [np.nan] * (n - len(vals))
    return np.array(vals[:n], dtype=float)

def pct_change_step(x):
    # %Δ step: 100*(x[i]-x[i-1])/x[i-1], first = 0
    p = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        if np.isfinite(x[i]) and np.isfinite(x[i-1]) and x[i-1] != 0:
            p[i] = 100.0 * (x[i] - x[i-1]) / x[i-1]
        else:
            p[i] = 0.0
    return p

def pct_change_window(x, w):
    # window %Δ: 100*(x[i]-x[i-w])/x[i-w], for i<w => 0
    p = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        if i - w >= 0 and np.isfinite(x[i]) and np.isfinite(x[i-w]) and x[i-w] != 0:
            p[i] = 100.0 * (x[i] - x[i-w]) / x[i-w]
        else:
            p[i] = 0.0
    return p

def clamp01(a):
    return np.clip(a, -10, 10)  # just avoid crazy values

def lorentz_E(T):
    return 1.0 - (T ** 2)

def v_of(series):
    v = np.zeros_like(series, dtype=float)
    for i in range(1, len(series)):
        v[i] = series[i] - series[i-1]
    return v

def vshape_filter(is_drop, recover, max_gap=2):
    """
    If we see a drop at i and recovery within <=max_gap steps, mark INFO (false-alarm-like).
    return boolean array info_mask
    """
    n = len(is_drop)
    info = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_drop[i]:
            jmax = min(n - 1, i + max_gap)
            if recover[i:jmax+1].any():
                info[i] = True
    return info

def level_from_T(T_abs, warn_thr, red_thr):
    # returns 0=GREEN,1=WARNING,2=RED per point
    lvl = np.zeros_like(T_abs, dtype=int)
    lvl[T_abs >= warn_thr] = 1
    lvl[T_abs >= red_thr] = 2
    return lvl

def combine_gatekeeper(lvl_hrv, lvl_rr, lvl_spo2):
    # DN RED if >=2 systems are RED at minute t
    # DN WARNING if >=2 systems are WARNING/RED at minute t
    dn = np.zeros_like(lvl_hrv, dtype=int)
    for i in range(len(dn)):
        reds = int(lvl_hrv[i] == 2) + int(lvl_rr[i] == 2) + int(lvl_spo2[i] == 2)
        warns_or_reds = int(lvl_hrv[i] >= 1) + int(lvl_rr[i] >= 1) + int(lvl_spo2[i] >= 1)
        if reds >= 2:
            dn[i] = 2
        elif warns_or_reds >= 2:
            dn[i] = 1
        else:
            dn[i] = 0
    return dn

def color_for_dn(dn):
    # 0 green,1 yellow,2 red
    return np.where(dn==2, "red", np.where(dn==1, "gold", "green"))

# ----------------------------
# DN logic (spec-aligned)
# ----------------------------
def compute_dn(hrv, rr, spo2, W=5):
    n = len(hrv)
    minutes = np.arange(1, n+1)

    # ----- HRV (baseline-free, shape-based) -----
    # Use %ΔHRV step for TT + E + vT/vE
    p_hrv = pct_change_step(hrv)  # %
    T_hrv = clamp01(p_hrv / 80.0)   # per your ET/DN dynamic convention
    E_hrv = lorentz_E(T_hrv)
    vT_hrv = v_of(T_hrv)
    vE_hrv = v_of(E_hrv)

    # HRV flags (Core v1 style):
    # Step-drop RED if any step <= -40%
    hrv_step_drop_red = (p_hrv <= -40.0)

    # "drift" on HRV using window drop (cumulative) — still baseline-free (local window)
    p_hrv_w = pct_change_window(hrv, W)  # if negative => drop across W mins
    # treat negative cumulative drop magnitude
    T_hrv_w = clamp01((-p_hrv_w) / 80.0)  # drop -> positive "badness"
    # Severity for HRV:
    # RED if step-drop event OR window-drop |T| >= 0.6 (≈48% drop across W, strong)
    # WARNING if window-drop |T| >= 0.3 (≈24% drop across W)
    hrv_bad_T = np.maximum(np.abs(T_hrv_w), np.abs(T_hrv))  # keep for debug
    lvl_hrv = np.zeros(n, dtype=int)
    lvl_hrv[np.abs(T_hrv_w) >= 0.30] = 1
    lvl_hrv[np.abs(T_hrv_w) >= 0.60] = 2
    lvl_hrv[hrv_step_drop_red] = 2

    # HRV V-shape: if big drop then recovery quickly => INFO (downgrade to GREEN unless other evidence)
    # define drop as p<=-25% and recovery as p>=+15%
    hrv_drop = (p_hrv <= -25.0)
    hrv_recover = (p_hrv >= +15.0)
    hrv_info = vshape_filter(hrv_drop, hrv_recover, max_gap=2)
    # downgrade WARNING->GREEN on V-shape only (keep RED if strong)
    lvl_hrv[(lvl_hrv == 1) & hrv_info] = 0

    # ----- RR (k=25) -----
    p_rr_step = pct_change_step(rr)
    p_rr_w = pct_change_window(rr, W)

    T_rr_step = clamp01(p_rr_step / 25.0)
    T_rr_w = clamp01(p_rr_w / 25.0)

    # badness uses BOTH: SHOCK(step) + DRIFT(window)
    T_rr_bad = np.maximum(T_rr_step, T_rr_w)
    # thresholds per your spec:
    # RED if |T|>=1, WARNING if |T|>=0.5
    lvl_rr = level_from_T(np.abs(T_rr_bad), warn_thr=0.5, red_thr=1.0)

    # RR V-shape filter (artifact): spike up then back down quickly => INFO
    rr_spike = (p_rr_step >= +20.0)
    rr_relief = (p_rr_step <= -15.0)
    rr_info = vshape_filter(rr_spike, rr_relief, max_gap=2)
    lvl_rr[(lvl_rr == 1) & rr_info] = 0

    # ----- SpO2 (k=5) -----
    # We treat drop as bad: if spo2 decreases, p is negative -> convert to positive badness
    p_sp_step = pct_change_step(spo2)
    p_sp_w = pct_change_window(spo2, W)

    T_sp_step = clamp01((-p_sp_step) / 5.0)  # drop -> positive
    T_sp_w = clamp01((-p_sp_w) / 5.0)        # cumulative drop in window

    T_sp_bad = np.maximum(T_sp_step, T_sp_w)

    # thresholds per your spec:
    # RED if |T|>=0.6, WARNING if |T|>=0.3
    lvl_sp = level_from_T(np.abs(T_sp_bad), warn_thr=0.3, red_thr=0.6)

    # SpO2 V-shape: drop then immediate recovery => INFO
    sp_drop = (p_sp_step <= -2.0)   # >=2% drop in one step
    sp_recover = (p_sp_step >= +1.0)
    sp_info = vshape_filter(sp_drop, sp_recover, max_gap=2)
    lvl_sp[(lvl_sp == 1) & sp_info] = 0

    # ----- Gatekeeper DN (2 systems rule) -----
    dn = combine_gatekeeper(lvl_hrv, lvl_rr, lvl_sp)

    df = pd.DataFrame({
        "minute": minutes,
        "HRV": hrv, "RR": rr, "SpO2": spo2,
        "pHRV_step(%)": p_hrv, "T_HRV_step": T_hrv, "T_HRV_win": T_hrv_w, "vT_HRV": vT_hrv, "vE_HRV": vE_hrv, "lvl_HRV": lvl_hrv,
        "pRR_step(%)": p_rr_step, "pRR_win(%)": p_rr_w, "T_RR_step": T_rr_step, "T_RR_win": T_rr_w, "lvl_RR": lvl_rr,
        "pSp_step(%)": p_sp_step, "pSp_win(%)": p_sp_w, "T_Sp_step": T_sp_step, "T_Sp_win": T_sp_w, "lvl_SpO2": lvl_sp,
        "DN": dn,
    })
    return df

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="DN alert demo", layout="wide")
st.title("DN alert demo")
st.caption("Gatekeeper: DN RED if ≥2 systems are RED, DN WARNING if ≥2 systems are WARNING/RED. HRV is baseline-free (shape/drift), RR uses k=25, SpO₂ uses k=5.")

if "df" not in st.session_state:
    st.session_state.df = None
if "play_stop" not in st.session_state:
    st.session_state.play_stop = False

colL, colR = st.columns([1, 1])

with colL:
    st.subheader("Input (15 points each)")
    hrv_txt = st.text_area("HRV (ms)", value="42 41 40 39 38 37 36 35 34 33 32 31 30 29 28", height=80)
    rr_txt  = st.text_area("RR (breaths/min)", value="16 16 17 17 18 18 19 19 20 20 21 21 22 22 23", height=80)
    sp_txt  = st.text_area("SpO₂ (%)", value="98 98 97 97 96 96 95 95 94 94 93 93 92 92 91", height=80)

    W = st.selectbox("Drift window W (minutes)", options=[3,5], index=1)

    c1, c2, c3 = st.columns(3)
    run_static = c1.button("Run static")
    play = c2.button("Play (3s/step)")
    stop = c3.button("Stop")

    if stop:
        st.session_state.play_stop = True

def render_plot(df, upto=None):
    if upto is None:
        dfx = df.copy()
    else:
        dfx = df.iloc[:upto].copy()

    x = dfx["minute"].values
    y = dfx["DN"].values
    colors = color_for_dn(y)

    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("DN state per minute (0=GREEN, 1=WARNING, 2=RED)")
    ax.set_xlabel("Minute")
    ax.set_ylabel("DN state")
    ax.set_xlim(1, 15)
    ax.set_ylim(-0.1, 2.1)
    ax.set_xticks(np.arange(1, 16, 1))
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(["GREEN","WARNING","RED"])
    ax.grid(True, alpha=0.25)

    # line + colored points
    ax.plot(x, y, linewidth=1)
    ax.scatter(x, y, s=40, c=colors)

    # horizontal lines for readability
    ax.axhline(1, linestyle="--", linewidth=1, alpha=0.3)
    ax.axhline(2, linestyle="--", linewidth=1, alpha=0.3)
    return fig

with colR:
    st.subheader("Output")

    if run_static or play:
        hrv = parse_series(hrv_txt, 15)
        rr  = parse_series(rr_txt, 15)
        sp  = parse_series(sp_txt, 15)
        df = compute_dn(hrv, rr, sp, W=W)
        st.session_state.df = df
        st.session_state.play_stop = False

    df = st.session_state.df

    if df is None:
        st.info("Press **Run static** to compute DN (no auto-run on edit).")
    else:
        red_minutes = df.loc[df["DN"]==2, "minute"].tolist()
        warn_minutes = df.loc[df["DN"]==1, "minute"].tolist()

        st.write(f"**DN RED minutes:** {red_minutes if red_minutes else 'None'}")
        st.write(f"**DN WARNING minutes:** {warn_minutes if warn_minutes else 'None'}")

        if play and (not st.session_state.play_stop):
            placeholder = st.empty()
            for i in range(1, 16):
                if st.session_state.play_stop:
                    break
                fig = render_plot(df, upto=i)
                placeholder.pyplot(fig, clear_figure=True)
                time.sleep(3)
        else:
            fig = render_plot(df)
            st.pyplot(fig)

        with st.expander("Details (per minute)", expanded=False):
            show_cols = [
                "minute",
                "lvl_HRV","lvl_RR","lvl_SpO2","DN",
                "pHRV_step(%)","T_HRV_win","vT_HRV","vE_HRV",
                "pRR_step(%)","pRR_win(%)","T_RR_win",
                "pSp_step(%)","pSp_win(%)","T_Sp_win",
            ]
            st.dataframe(df[show_cols], use_container_width=True)
