import os
import io
import hashlib
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from lifelines import CoxPHFitter
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# === LLM: safe import of OpenAI SDK ===========================================
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_AVAILABLE = False

# ────────────────────────── Settings / Branding ──────────────────────────
PAGE_TITLE = "ElderCare Risk Demo"
PAGE_ICON = "🩺"
HIDE_SYSTEM_UI = False
SHOW_HEADER_LOGO = False

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# === LLM: read secrets/environment for API key + model =========================
# Prefer Streamlit secrets; fall back to environment variables.
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

def get_openai_client() -> Tuple[Optional["OpenAI"], Optional[str]]:
    """Return an OpenAI client or (None, error_message) if not available."""
    if not OPENAI_AVAILABLE:
        return None, "OpenAI SDK not installed. Run: pip install openai>=1.30"
    if not OPENAI_API_KEY:
        return None, "Missing OPENAI_API_KEY (set in .streamlit/secrets.toml or env)."
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client, None
    except Exception as e:
        return None, f"OpenAI init error: {e}"

LOGO_PATH = Path("logo.png")

if HIDE_SYSTEM_UI:
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Global CSS (pink/white)
st.markdown("""
<style>
.header {
  background: linear-gradient(90deg, #ec4899 0%, #f472b6 100%);
  color: white;
  padding: 14px 18px;
  border-radius: 14px;
  margin-top: 44px;
  margin-bottom: 14px;
}
h1, h2, h3, h4 { color: #111827; }
.block-container { padding-top: 1rem; }
.card {
  background: #fff;
  border: 1px solid #fce7f3;
  border-radius: 14px; padding: 14px; margin-bottom: 12px;
  box-shadow: 0 1px 0 rgba(236,72,153,0.05), 0 6px 20px rgba(236,72,153,0.06);
}
.stButton>button { background:#ec4899; color:white; border-radius:10px; border:none; }
.stButton>button:hover { opacity:.92; }
[data-testid="stDataFrame"] .st-emotion-cache-1yycg5l { background-color:#fff1f7; }
.badge {display:inline-block;border-radius:8px;padding:4px 8px;font-weight:600;color:#fff;background:#ec4899;}
.kpi {font-weight:700;}
</style>
""", unsafe_allow_html=True)

# Header
hcol1, hcol2 = st.columns([1,5])
with hcol1:
    if SHOW_HEADER_LOGO and LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
with hcol2:
    st.markdown(f"""
    <div class="header">
      <div style="font-size:22px;font-weight:700;">{PAGE_TITLE}</div>
      <div style="opacity:.95;">Demo — synthetic data. Pink &amp; white theme.</div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    st.markdown("### ElderCare Demo")
    st.caption("Synthetic data — pink/white theme")
    page = st.radio("Page", ["Worklist", "Facility Dashboard"], index=0, horizontal=False)

# Plotly theme
pio.templates.default = "plotly_white"
BRAND_PINK = "#ec4899"
BRAND_ORANGE = "#f59e0b"
BRAND_GREEN = "#10b981"

# ────────────────────────── Data Loading ──────────────────────────
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    patients = pd.read_csv("patients_features.csv", parse_dates=["date_last"])
    ts = pd.read_csv("patients_timeseries.csv", parse_dates=["date"])
    if "facility" not in patients.columns:
        facilities = ["Central Clinic", "Lakeside Care", "Sunset Manor", "North Home"]
        def assign_facility(pid: str) -> str:
            h = int(hashlib.md5(pid.encode()).hexdigest(), 16)
            return facilities[h % len(facilities)]
        patients["facility"] = patients["patient_id"].apply(assign_facility)
    return patients, ts

patients, ts = load_data()

# ────────────────────────── Features & Explanations ──────────────────────────
FEATURES = [
    "age","comorbidity_idx","frailty_cfs","acb_score","baseline_mmse",
    "sleep_mean_7d","sleep_slope_7d","steps_mean_7d","steps_slope_7d",
    "spo2_last","spo2_slope_7d","mood_last","mood_slope_7d",
    "adherence_mean_7d","adherence_low_days",
    "puzzle_acc_last","reaction_ms_last","orientation_last","orientation_slope_7d"
]

INDICATOR_EXPLAIN = {
    "sleep_mean_7d": "Average hours slept in the last 7 days.",
    "sleep_slope_7d": "Sleep trend over the last week (positive = improving).",
    "steps_mean_7d": "Average daily steps in the last 7 days.",
    "steps_slope_7d": "Steps trend (positive = increasing activity).",
    "spo2_last": "Most recent oxygen saturation (%).",
    "spo2_slope_7d": "SpO₂ trend (negative = worsening).",
    "mood_last": "Latest self-reported mood (1–5).",
    "mood_slope_7d": "Mood trend (positive = improving).",
    "adherence_mean_7d": "Medication adherence (0–1) in last week.",
    "adherence_low_days": "Days in last week with adherence < 0.6.",
    "puzzle_acc_last": "Most recent cognitive micro-task accuracy (0–1).",
    "reaction_ms_last": "Most recent reaction time (ms).",
    "orientation_last": "Latest orientation score (0–1).",
    "orientation_slope_7d": "Orientation trend (positive = improving).",
    "age": "Age in years.",
    "comorbidity_idx": "Comorbidity burden.",
    "frailty_cfs": "Clinical Frailty Scale (1–7).",
    "acb_score": "Anticholinergic Cognitive Burden score.",
    "baseline_mmse": "Baseline MMSE (0–30).",
}

# ────────────────────────── Models ──────────────────────────
@st.cache_resource
def train_models(df: pd.DataFrame):
    rng = np.random.default_rng(0)
    ids = df["patient_id"].unique()
    rng.shuffle(ids)
    split = int(0.75 * len(ids))
    train_ids = set(ids[:split])
    train = df[df.patient_id.isin(train_ids)].copy()
    test  = df[~df.patient_id.isin(train_ids)].copy()

    logit_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, penalty="l1", solver="liblinear", class_weight="balanced")),
    ])
    logit_pipe.fit(train[FEATURES], train["hosp_30d"].astype(int))

    cal = CalibratedClassifierCV(logit_pipe, cv=3, method="isotonic")
    cal.fit(train[FEATURES], train["hosp_30d"].astype(int))

    surv_cols = FEATURES
    df_surv_tr = train[surv_cols + ["death_within_180d","death_time_days","patient_id"]].copy()
    df_surv_tr["event"] = (df_surv_tr["death_within_180d"] == 1).astype(int)
    df_surv_tr["duration"] = np.minimum(df_surv_tr["death_time_days"], 180)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(
        df_surv_tr.drop(columns=["patient_id","death_within_180d","death_time_days"]),
        duration_col="duration",
        event_col="event",
    )

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(train[FEATURES], train["dem_prog_6m"].astype(int))

    return cal, logit_pipe, cph, gb, train, test

cal, logit_pipe, cph, gb, train, test = train_models(patients)

# Score everyone
patients = patients.copy()
patients["risk_hosp30"] = cal.predict_proba(patients[FEATURES])[:, 1]
haz = cph.predict_partial_hazard(patients[FEATURES]).values
haz = (haz - haz.min()) / (haz.max() - haz.min() + 1e-9)
patients["risk_surv6m"] = haz
patients["risk_demprog6m"] = gb.predict_proba(patients[FEATURES])[:, 1]

# Buckets
q90 = patients["risk_hosp30"].quantile(0.9)
q70 = patients["risk_hosp30"].quantile(0.7)
def bucket(p: float) -> str:
    return "High" if p >= q90 else ("Medium" if p >= q70 else "Low")
patients["risk_bucket"] = patients["risk_hosp30"].apply(bucket)

# Drivers (coef * standardized value)
scaler_for_show = StandardScaler().fit(patients[FEATURES])
X_scaled = scaler_for_show.transform(patients[FEATURES])
coef = logit_pipe.named_steps["clf"].coef_[0]
contrib = X_scaled * coef
topk_idx = np.argsort(-np.abs(contrib), axis=1)[:, :5]
top_features = np.array(FEATURES)[topk_idx]
top_values = np.take_along_axis(contrib, topk_idx, axis=1)
patients["top_drivers"] = [
    "; ".join([f"{top_features[i, j]} ({top_values[i, j]:+.2f})" for j in range(5)])
    for i in range(patients.shape[0])
]

# Helpers
def bucket_color(b: str) -> str:
    return {"High": "#ec4899", "Medium": "#f59e0b", "Low": "#10b981"}.get(b, "#6b7280")

def ts_card(df: pd.DataFrame, y: str, title: str, yaxis: str):
    fig = px.line(df, x="date", y=y, title=title, color_discrete_sequence=["#ec4899"])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=250, yaxis_title=yaxis, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

def make_narrative(r: pd.Series) -> str:
    parts = []
    if r["risk_bucket"] == "High":
        parts.append("Priority review within 24 hours due to elevated 30-day hospitalization risk.")
    elif r["risk_bucket"] == "Medium":
        parts.append("Monitor closely; risk is elevated compared to peers.")
    else:
        parts.append("Risk appears low; continue routine monitoring.")
    if r["orientation_last"] < 0.7:
        parts.append("Orientation is low; consider reality-orientation checks and look for delirium triggers.")
    if (r["adherence_mean_7d"] < 0.7) or (r["adherence_low_days"] >= 2):
        parts.append("Medication adherence is suboptimal; verify regimen and caregiver prompts.")
    if r["spo2_last"] < 92:
        parts.append("SpO₂ < 92%; recheck and escalate per protocol.")
    if r["sleep_mean_7d"] < 5.5:
        parts.append("Sleep is poor; consider sleep hygiene and daytime activity.")
    return " ".join(parts)

# PDFs
def build_patient_pdf(r: pd.Series, pt_ts_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 0.75 * inch
    y = height - margin
    def line(txt, size=11, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(margin, y, txt); y -= 14
    c.setFont("Helvetica-Bold", 16); c.drawString(margin, y, "Patient Handoff"); y -= 22
    c.setFont("Helvetica", 10); c.drawString(margin, y, "Synthetic demo — not for clinical use"); y -= 18
    line(f"Patient ID: {r['patient_id']}", 12, True)
    line(f"Facility: {r['facility']}  |  Age: {int(r['age'])}  |  Sex: {r['sex']}")
    line(f"Comorbidity Index: {int(r['comorbidity_idx'])}  |  CFS: {r['frailty_cfs']}  |  MMSE: {int(r['baseline_mmse'])}")
    y -= 6
    line(f"30d Hosp Risk: {r['risk_hosp30']*100:.1f}%  |  6m Survival Risk*: {r['risk_surv6m']*100:.1f}%  |  6m Dementia Prog: {r['risk_demprog6m']*100:.1f}%", 11, True)
    line(f"Bucket: {r['risk_bucket']}")
    y -= 6
    line("Top Drivers:", 12, True)
    for d in str(r["top_drivers"]).split("; ")[:5]:
        line(f"- {d}")
    y -= 6
    line("Nurse Summary:", 12, True)
    import textwrap as tw
    for seg in tw.wrap(make_narrative(r), width=95):
        line(seg)
    y -= 6
    line("Recent indicators:", 12, True)
    last = pt_ts_df.iloc[-1] if len(pt_ts_df) > 0 else None
    if last is not None:
        metrics = [
            f"Sleep: {last['sleep_h']:.1f} h",
            f"Steps: {int(last['steps'])}",
            f"SpO₂: {last['spo2']:.1f}%",
            f"Mood: {last['mood_1to5']:.1f}/5",
            f"Orientation: {last['orientation_pct']:.2f}",
            f"Adherence (7d avg): {r['adherence_mean_7d']:.2f}",
        ]
        for m in metrics: line(f"- {m}")
    else:
        line("- No recent time series available")
    c.showPage(); c.save()
    pdf = buf.getvalue(); buf.close(); return pdf

def build_facility_pdf(fac_name: str, df_fac: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 0.75 * inch
    y = height - margin
    def line(txt, size=11, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(margin, y, txt); y -= 14
    c.setFont("Helvetica-Bold", 16); c.drawString(margin, y, f"Facility Report — {fac_name}"); y -= 22
    c.setFont("Helvetica", 10); c.drawString(margin, y, "Synthetic demo — not for clinical use"); y -= 18
    total = len(df_fac)
    high = int((df_fac["risk_bucket"] == "High").sum())
    med = int((df_fac["risk_bucket"] == "Medium").sum())
    low = int((df_fac["risk_bucket"] == "Low").sum())
    line(f"Total patients: {total}  |  High: {high}  |  Medium: {med}  |  Low: {low}", 12, True)
    y -= 8
    df_sorted = df_fac.sort_values("risk_hosp30", ascending=False).copy()
    for _, r in df_sorted.iterrows():
        if y < 120: c.showPage(); y = height - margin
        line(f"{r['patient_id']} — 30d Hosp Risk {r['risk_hosp30']*100:.1f}% ({r['risk_bucket']})", 12, True)
        line(f"Age {int(r['age'])}, Sex {r['sex']}, CFS {r['frailty_cfs']}, MMSE {int(r['baseline_mmse'])}")
        drivers = "; ".join(str(r["top_drivers"]).split("; ")[:2])
        line(f"Top drivers: {drivers}")
        import textwrap as tw
        for seg in tw.wrap(make_narrative(r), width=95):
            line(seg)
        y -= 8
    c.showPage(); c.save()
    pdf = buf.getvalue(); buf.close(); return pdf

# ────────────────────────── STATE & Alerts ──────────────────────────
if "reviewed" not in st.session_state: st.session_state.reviewed = {}
if "notes" not in st.session_state: st.session_state.notes = {}
if "jump_to_pid" not in st.session_state: st.session_state.jump_to_pid = None

def alerts_for_row(r: pd.Series):
    msgs = []
    if r["spo2_last"] < 92: msgs.append("SpO₂ below 92%")
    if r["orientation_last"] < 0.6 or r["orientation_slope_7d"] < -0.15: msgs.append("Orientation decline")
    if r["adherence_low_days"] >= 3: msgs.append("Medication adherence low (≥3 low days)")
    if r["sleep_mean_7d"] < 5: msgs.append("Poor sleep (avg < 5h)")
    if r["risk_bucket"] == "High": msgs.append("High 30d hospitalization risk")
    return msgs

# === LLM: prompts & call =======================================================
def build_facility_summary_prompt(fac_name: str, df_fac: pd.DataFrame) -> str:
    total = len(df_fac)
    high = int((df_fac["risk_bucket"] == "High").sum())
    med = int((df_fac["risk_bucket"] == "Medium").sum())
    low = int((df_fac["risk_bucket"] == "Low").sum())
    mean_hosp = df_fac["risk_hosp30"].mean()*100
    alerts_map = {}
    for _, r in df_fac.iterrows():
        pid = r["patient_id"]; alerts_map[pid] = alerts_for_row(r)
    top_patients = df_fac.sort_values("risk_hosp30", ascending=False).head(5)[
        ["patient_id","risk_hosp30","risk_bucket","frailty_cfs","spo2_last","adherence_mean_7d","orientation_last"]
    ]
    lines = [f"Facility: {fac_name}",
             f"Counts — total:{total}, high:{high}, medium:{med}, low:{low}",
             f"Mean 30d hospitalization risk: {mean_hosp:.1f}%",
             "Top patients by 30d risk:"]
    for _, row in top_patients.iterrows():
        lines.append(
            f"- {row['patient_id']}: {row['risk_hosp30']*100:.1f}% ({row['risk_bucket']}), "
            f"CFS {row['frailty_cfs']}, SpO2 {row['spo2_last']:.1f}%, "
            f"adherence {row['adherence_mean_7d']:.2f}, orientation {row['orientation_last']:.2f}"
        )
    lines.append("Alerts per patient (selected):")
    for pid, alerts in list(alerts_map.items())[:10]:
        if alerts:
            lines.append(f"- {pid}: {', '.join(alerts)}")
    return "\n".join(lines)

def build_patient_summary_prompt(r: pd.Series) -> str:
    return f"""Create a short, clinical but plain-language summary for a geriatric nurse huddle.
Patient {r['patient_id']} (age {int(r['age'])}, sex {r['sex']}, CFS {r['frailty_cfs']}, MMSE {int(r['baseline_mmse'])}).
Risks: 30d hospitalization {r['risk_hosp30']*100:.1f}%, 6m survival risk index {r['risk_surv6m']*100:.1f}%, 6m dementia progression {r['risk_demprog6m']*100:.1f}%.
Top drivers: {str(r['top_drivers'])}.
Recent: SpO2 {r['spo2_last']:.1f}%, sleep mean(7d) {r['sleep_mean_7d']:.1f}h, adherence {r['adherence_mean_7d']:.2f}, orientation {r['orientation_last']:.2f}.

Write 3–5 bullet points: condition/concern, likely drivers, suggested checks or next steps (vitals, meds review, orientation activities, safety). Keep it under 300 words. Avoid alarmist tone. Reader needs good explanation, assume reader is not technical. Do not say anything else like an intro or suggested follow up, the output needs to be ready to be printed.
"""

def call_llm(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    client, err = get_openai_client()
    if err: return None, err
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        if resp and getattr(resp, "output_text", None):
            return resp.output_text, None
        # Fallback for older response shapes
        try:
            return resp.choices[0].message.content, None  # type: ignore
        except Exception:
            return None, "Unexpected OpenAI response format."
    except Exception as e:
        return None, f"OpenAI error: {e}"

# ────────────────────────── PAGES ──────────────────────────

# Worklist Page
if page == "Worklist":
    st.sidebar.title("Filters")
    fac_opts = sorted(patients["facility"].unique())
    facility_filter = st.sidebar.multiselect("Facility", options=fac_opts, default=fac_opts)
    sex_filter = st.sidebar.multiselect("Sex", options=sorted(patients["sex"].unique()), default=sorted(patients["sex"].unique()))
    bucket_filter = st.sidebar.multiselect("Risk bucket", options=["High","Medium","Low"], default=["High","Medium","Low"])
    age_min, age_max = int(patients["age"].min()), int(patients["age"].max())
    age_sel = st.sidebar.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    # Cohort chips
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Quick cohorts")
    colA, colB, colC, colD = st.columns(4)
    with colA: chip_frail = st.toggle("Frailty ≥ 6")
    with colB: chip_lowadh = st.toggle("Low adherence")
    with colC: chip_spo2drop = st.toggle("SpO₂ drop (7d)")
    with colD: chip_newhigh = st.toggle("New High-risk")

    base = patients[
        (patients.facility.isin(facility_filter)) &
        (patients.sex.isin(sex_filter)) &
        (patients.risk_bucket.isin(bucket_filter)) &
        (patients.age.between(age_sel[0], age_sel[1]))
    ].copy()

    mask = pd.Series(True, index=base.index)
    if chip_frail: mask &= base["frailty_cfs"] >= 6
    if chip_lowadh: mask &= (base["adherence_mean_7d"] < 0.7) | (base["adherence_low_days"] >= 2)
    if chip_spo2drop: mask &= base["spo2_slope_7d"] < -0.2
    if chip_newhigh:
        mask &= (base["risk_bucket"] == "High") & ((pd.Timestamp.today().normalize() - base["date_last"]) <= pd.Timedelta(days=7))
    filtered = base[mask].copy()
    st.caption("Toggle chips to focus cohorts (you can combine them).")
    st.markdown('</div>', unsafe_allow_html=True)

    # Alerts panel (per patient)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔔 Alerts (today’s escalations)")
    alerts_by_pid = {}
    for _, r in filtered.iterrows():
        msgs = alerts_for_row(r)
        if msgs:
            pid = r["patient_id"]
            alerts_by_pid[pid] = sorted(set(alerts_by_pid.get(pid, []) + msgs))
    if alerts_by_pid:
        for i, (pid, msgs) in enumerate(alerts_by_pid.items(), 1):
            c1, c2 = st.columns([5,1])
            c1.markdown(f"**{i}. Patient {pid}** — " + "; ".join(msgs))
            if c2.button("Go to patient", key=f"go_{pid}_{i}"):
                st.session_state.jump_to_pid = pid
    else:
        st.caption("No escalations in the current filter set.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Worklist table with badges
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏥 Nurse Worklist — Risk (Synthetic)")
    st.caption("Traffic-light: High (pink) → review in 24h; Medium (amber) → watchlist; Low (green) → routine.")
    filtered["Reviewed"] = filtered["patient_id"].map(lambda pid: "✅" if st.session_state.reviewed.get(pid, False) else "")
    filtered["Notes"] = filtered["patient_id"].map(lambda pid: len(st.session_state.notes.get(pid, [])))
    alerts_count_map = {r["patient_id"]: len(alerts_for_row(r)) for _, r in filtered.iterrows()}
    filtered["Alerts"] = filtered["patient_id"].map(lambda pid: alerts_count_map.get(pid, 0))
    disp = filtered[[
        "facility","patient_id","age","sex","comorbidity_idx","frailty_cfs","baseline_mmse",
        "risk_hosp30","risk_surv6m","risk_demprog6m","risk_bucket","Alerts","Reviewed","Notes"
    ]].copy()
    disp["risk_hosp30"] = (disp["risk_hosp30"]*100).round(1)
    disp["risk_surv6m"] = (disp["risk_surv6m"]*100).round(1)
    disp["risk_demprog6m"] = (disp["risk_demprog6m"]*100).round(1)
    st.dataframe(disp.sort_values(["facility","risk_hosp30"], ascending=[True, False]), use_container_width=True)
    csv_data = disp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download current worklist (CSV)", data=csv_data, file_name="worklist_filtered.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

    # Patient details
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Patient details")
    if len(disp) == 0:
        st.info("No patients match the current filters/chips.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        options = disp.sort_values("risk_hosp30", ascending=False)["patient_id"].unique().tolist()
        if st.session_state.jump_to_pid in options:
            default_index = options.index(st.session_state.jump_to_pid)
            st.session_state.jump_to_pid = None
        else:
            default_index = 0
        pid = st.selectbox("Choose patient", options=options, index=default_index)
        row = patients[patients.patient_id == pid].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("30d Hosp. Risk", f"{row['risk_hosp30']*100:.1f}%")
        c2.metric("6m Survival Risk*", f"{row['risk_surv6m']*100:.1f}%")
        c3.metric("6m Dementia Prog.", f"{row['risk_demprog6m']*100:.1f}%")
        c4.markdown(
            f"**Bucket:** <span style='background-color:{bucket_color(row['risk_bucket'])};"
            f"color:white;padding:0.35rem 0.6rem;border-radius:8px'>{row['risk_bucket']}</span>",
            unsafe_allow_html=True
        )

        st.subheader("Nurse Summary (rule-based)")
        st.write(make_narrative(row))

        # === LLM: Patient AI Summary =========================================
        st.subheader("AI Summary (patient)")
        if st.button("🧠 Generate AI summary for this patient"):
            prompt = build_patient_summary_prompt(row)
            text, err = call_llm(prompt)
            if err:
                st.info(f"LLM not available: {err}")
            else:
                st.text_area("AI summary", value=text, height=150)
                st.download_button("⬇️ Download AI summary (txt)", data=text.encode("utf-8"),
                                   file_name=f"{pid}_ai_summary.txt", mime="text/plain")

        st.markdown('</div>', unsafe_allow_html=True)

        # Charts
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Last 90 days (time series)")
        pt_ts = ts[ts.patient_id == pid].sort_values("date").tail(90)
        cc1, cc2 = st.columns(2)
        with cc1:
            ts_card(pt_ts, "sleep_h", "Sleep (hours)", "hours")
            ts_card(pt_ts, "spo2", "SpO₂ (%)", "%")
            ts_card(pt_ts, "puzzle_acc_0to1", "Puzzle accuracy", "0–1")
        with cc2:
            ts_card(pt_ts, "steps", "Steps (daily)", "steps")
            ts_card(pt_ts, "mood_1to5", "Mood (1–5)", "score")
            ts_card(pt_ts, "orientation_pct", "Orientation", "0–1")
        st.markdown('</div>', unsafe_allow_html=True)

        # Indicator explanations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.expander("What do these indicators mean?"):
            for k, v in INDICATOR_EXPLAIN.items():
                st.markdown(f"**{k}** — {v}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Patient PDF
        pt_ts = ts[ts.patient_id == pid].sort_values("date").tail(90)
        pdf_bytes = build_patient_pdf(row, pt_ts)
        st.download_button("🖨️ Download patient handoff (PDF)", data=pdf_bytes, file_name=f"{pid}_handoff.pdf", mime="application/pdf")

        # Fake chat transcript
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Fake chat transcript (demo sentiment/agitation)")
        def gen_fake_chat(pid: str, risk: float, days: int = 7, seed: int = 0):
            rng = np.random.default_rng(abs(int(hashlib.md5((pid+str(seed)).encode()).hexdigest(), 16)) % (2**32))
            rows = []
            base_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
            p_agitated = min(0.05 + 0.6*risk, 0.8)
            for d in range(days):
                day = base_date + pd.Timedelta(days=d)
                n_msgs = rng.integers(3, 7)
                for _ in range(n_msgs):
                    t = day + pd.Timedelta(minutes=int(rng.integers(7, 1200)))
                    rdraw = rng.random()
                    if rdraw < 0.15:
                        sent = "positive"; text = rng.choice(["Feeling okay today.","Did my exercises.","Enjoyed music therapy.","Slept well last night."])
                    elif rdraw < 0.4:
                        sent = "neutral"; text = rng.choice(["Just finished lunch.","I took my pills.","Walked a bit.","I’m resting now."])
                    elif rdraw < 0.4 + p_agitated:
                        sent = "agitated"; text = rng.choice(["I can't find my things!","Leave me alone!","Why am I here?","I want to go home now!"])
                    else:
                        sent = "negative"; text = rng.choice(["I feel tired.","I didn't sleep much.","My head hurts.","I feel confused."])
                    rows.append({"timestamp": t, "patient_id": pid, "speaker": np.random.choice(["Patient","Assistant"]), "sentiment": sent, "text": text})
            chat = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            score = ((chat.sentiment=="positive").sum()*1 + (chat.sentiment=="neutral").sum()*0 + (chat.sentiment=="negative").sum()*(-1) + (chat.sentiment=="agitated").sum()*(-2)) / max(len(chat),1)
            return chat, score
        if "chat_seed" not in st.session_state: st.session_state.chat_seed = 0
        if st.button("Generate fake chat (last 7 days)"): st.session_state.chat_seed += 1
        chat_df, chat_score = gen_fake_chat(pid, float(row["risk_hosp30"]), days=7, seed=st.session_state.chat_seed)
        st.caption(f"Aggregate chat sentiment score (last 7 days): {chat_score:.2f}  (more negative → lower)")
        st.dataframe(chat_df, use_container_width=True, height=240)
        chat_csv = chat_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download chat transcript (CSV)", data=chat_csv, file_name=f"{pid}_chat_transcript.csv", mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

# Facility Dashboard
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏢 Facility overview")
    fac_grp = patients.groupby("facility").agg(
        total=("patient_id","count"),
        high=("risk_bucket", lambda s: (s=="High").sum()),
        medium=("risk_bucket", lambda s: (s=="Medium").sum()),
        low=("risk_bucket", lambda s: (s=="Low").sum()),
        mean_hosp30=("risk_hosp30","mean"),
    ).reset_index()
    fac_grp["pct_high"] = fac_grp["high"] / fac_grp["total"] * 100
    cA, cB, cC = st.columns(3)
    cA.metric("Total facilities", f"{fac_grp.shape[0]}")
    cB.metric("Patients (all facilities)", f"{patients.shape[0]}")
    cC.metric("High-risk patients", f"{int((patients['risk_bucket']=='High').sum())}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### % High-risk by facility")
    fig1 = px.bar(
        fac_grp.sort_values("pct_high", ascending=False),
        x="facility", y="pct_high",
        text=fac_grp.sort_values("pct_high", ascending=False)["pct_high"].round(1),
        color_discrete_sequence=["#ec4899"]
    )
    fig1.update_layout(yaxis_title="% High-risk", xaxis_title=None, height=360)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Patient counts by facility (High/Medium/Low)")
    fac_melt = fac_grp.melt(id_vars="facility", value_vars=["high","medium","low"], var_name="bucket", value_name="count")
    fac_melt["bucket"] = fac_melt["bucket"].str.capitalize()
    fig2 = px.bar(
        fac_melt, x="facility", y="count", color="bucket",
        barmode="stack",
        color_discrete_map={"High":"#ec4899","Medium":"#f59e0b","Low":"#10b981"}
    )
    fig2.update_layout(yaxis_title="Patients", xaxis_title=None, height=380, legend_title=None)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Synthetic trend
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 14-day trend — average 30d hospitalization risk (synthetic)")
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=14)
    trend_rows = []
    rng = np.random.default_rng(2024)
    for _, r in fac_grp.iterrows():
        base = r["mean_hosp30"]; noise = rng.normal(0, 0.02, size=len(dates))
        vals = np.clip(base + np.cumsum(noise)/5, 0, 1)
        for d, v in zip(dates, vals):
            trend_rows.append({"facility": r["facility"], "date": d, "avg_risk": v*100})
    trend_df = pd.DataFrame(trend_rows)
    fig3 = px.line(trend_df, x="date", y="avg_risk", color="facility", color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(yaxis_title="% avg risk", xaxis_title=None, height=420)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Drill-down + Facility PDF + LLM summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Facility drill-down, export & AI summary")
    fac_sel = st.selectbox("Choose a facility", options=sorted(patients["facility"].unique()))
    df_fac = patients[patients["facility"] == fac_sel].copy()
    tbl = df_fac[["patient_id","age","sex","frailty_cfs","baseline_mmse","risk_hosp30","risk_bucket"]].copy()
    tbl["risk_hosp30"] = (tbl["risk_hosp30"]*100).round(1)
    st.dataframe(tbl.sort_values("risk_hosp30", ascending=False), use_container_width=True, height=300)
    pdf_fac = build_facility_pdf(fac_sel, df_fac)
    st.download_button(f"🖨️ Download {fac_sel} report (PDF)", data=pdf_fac, file_name=f"{fac_sel.replace(' ','_')}_report.pdf", mime="application/pdf")

    # === LLM: Facility Daily Summary =========================================
    st.markdown("##### AI Daily Summary (facility)")
    if st.button("🧠 Generate AI daily summary for this facility"):
        prompt = build_facility_summary_prompt(fac_sel, df_fac)
        text, err = call_llm(prompt + "\n\nWrite a concise, actionable 6–10 line daily briefing for nurses/admins. Use bullets, include counts, highlight top 3 patients to review, and concrete follow-ups.")
        if err:
            st.info(f"LLM not available: {err}")
        else:
            st.text_area("AI daily summary", value=text, height=220)
            st.download_button("⬇️ Download facility AI summary (txt)", data=text.encode("utf-8"),
                               file_name=f"{fac_sel.replace(' ','_')}_ai_daily_summary.txt", mime="text/plain")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.caption("Synthetic data for demo only. Risks are illustrative; not for clinical use.")

