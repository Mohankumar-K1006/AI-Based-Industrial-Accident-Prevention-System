"""
dashboard.py - Streamlit analytics dashboard.
Reads alerts_log.csv only; does not touch the camera or model.
Run: streamlit run dashboard.py
"""

import os
import io
import time
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

from pathlib import Path

_ROOT      = Path(__file__).resolve().parent
LOG_CSV    = str(_ROOT / "alerts" / "alerts_log.csv")
REPORT_DIR = str(_ROOT / "reports")

st.set_page_config(page_title="Safety Dashboard", layout="wide", page_icon="🏭")
st.title("🏭 Industrial Accident Prevention — Safety Dashboard")


@st.cache_data(ttl=2)  # re-read CSV at most every 2 s
def load_data():
    if not os.path.exists(LOG_CSV):
        return pd.DataFrame()
    try:
        with open(LOG_CSV, "r", encoding="utf-8") as fh:  # fresh open each call
            df = pd.read_csv(fh)
        if df.empty:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        for col in ("zone", "clip_filename"):
            if col not in df.columns:
                df[col] = ""
        df["zone"]          = df["zone"].fillna("default")
        df["clip_filename"] = df["clip_filename"].fillna("")
        return df
    except Exception:
        return pd.DataFrame()


# Sidebar controls
with st.sidebar:
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Auto-refreshes every 4 s")

df_all = load_data()

if df_all.empty:
    st.info("No incidents logged yet. The dashboard will populate once DANGER events are detected.")
    time.sleep(4)
    st.rerun()

today      = date.today()
start_date = st.sidebar.date_input("From", value=today)
end_date   = st.sidebar.date_input("To",   value=today)

mask = (df_all["timestamp"].dt.date >= start_date) & (df_all["timestamp"].dt.date <= end_date)
df   = df_all[mask].copy()

if df.empty:
    st.info(f"No incidents between {start_date} and {end_date}.")
    st.stop()

# KPI metrics + bar chart
st.subheader("Violations by Zone")
zone_counts = df.groupby("zone").size().reset_index(name="count")
cols = st.columns(max(len(zone_counts), 1))
for i, row in zone_counts.iterrows():
    cols[i % len(cols)].metric(label=f"Zone: {row['zone']}", value=int(row["count"]))

fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
ax_bar.bar(zone_counts["zone"], zone_counts["count"], color="#e74c3c")
ax_bar.set_xlabel("Zone"); ax_bar.set_ylabel("Incidents")
plt.tight_layout()
st.pyplot(fig_bar)
plt.close(fig_bar)

# Hourly heatmap
st.subheader("Violations by Hour of Day")
df["hour"] = df["timestamp"].dt.hour
hourly     = df.groupby("hour").size().reindex(range(24), fill_value=0)

fig_h, ax_h = plt.subplots(figsize=(12, 2))
im = ax_h.imshow([hourly.values], aspect="auto", cmap="YlOrRd")
ax_h.set_xticks(range(24)); ax_h.set_xticklabels([f"{h:02d}" for h in range(24)])
ax_h.set_yticks([]); ax_h.set_title("Incident heat by hour")
fig_h.colorbar(im, ax=ax_h, orientation="horizontal", fraction=0.05, pad=0.4)
plt.tight_layout()
st.pyplot(fig_h)
plt.close(fig_h)

# Recent incidents table
st.subheader("Recent 20 Incidents")
recent = df.sort_values("timestamp", ascending=False).head(20)

for idx, row in recent.iterrows():
    c1, c2, c3 = st.columns([1, 3, 2])
    with c1:
        img_path = str(row.get("image", ""))
        # resolve relative paths against the project root
        if img_path and not os.path.isabs(img_path):
            img_path = str(_ROOT / img_path)
        if img_path and os.path.exists(img_path):
            st.image(img_path, width=100)
        else:
            st.caption("image missing")
    with c2:
        st.write(f"**{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}**")
        st.write(f"Zone: `{row['zone']}` | Frame: `{row['frame']}`")
    with c3:
        clip = str(row.get("clip_filename", "")).strip()
        resolved_clip = None
        if clip:
            if os.path.exists(clip):
                resolved_clip = clip
            elif not os.path.isabs(clip) and os.path.exists(str(_ROOT / clip)):
                resolved_clip = str(_ROOT / clip)
            elif os.path.exists(str(_ROOT / "clips" / os.path.basename(clip))):
                resolved_clip = str(_ROOT / "clips" / os.path.basename(clip))

        if resolved_clip and os.path.exists(resolved_clip) and os.path.getsize(resolved_clip) > 0:
            with open(resolved_clip, "rb") as fclip:
                clip_data = fclip.read()
            st.download_button(
                "📥 Download clip", clip_data,
                file_name=os.path.basename(resolved_clip),
                mime="video/mp4",
                key=f"dl_{idx}")
            with st.expander("▶️ Watch Incident Clip", expanded=False):
                st.video(clip_data)
        else:
            age_secs = (datetime.now() - row['timestamp']).total_seconds()
            if 0 <= age_secs <= 15:
                st.caption("⏳ Recording clip in progress (~5s)...")
            else:
                st.caption("No clip available")

# PDF export
st.divider()
if st.button("Export shift report as PDF"):
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts_str      = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"{ts_str}_shift_report.pdf")

    c = pdf_canvas.Canvas(report_path, pagesize=A4)
    pw, ph = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, ph - 2*cm, "Industrial Safety - Shift Report")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, ph - 3*cm,   f"Period: {start_date}  to  {end_date}")
    c.drawString(2*cm, ph - 3.6*cm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, ph - 5*cm, "Incidents by Zone:")
    c.setFont("Helvetica", 11)
    y = ph - 5.8*cm
    for _, zrow in zone_counts.iterrows():
        c.drawString(2.5*cm, y, f"  {zrow['zone']}: {int(zrow['count'])}")
        y -= 0.6*cm

    top5 = recent.head(5)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y - 0.5*cm, "5 Most Recent Incidents:")
    y -= 1.5*cm
    for _, row in top5.iterrows():
        img_path = str(row.get("image", ""))
        ts_label = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, y, f"{ts_label}  |  Zone: {row['zone']}")
        if img_path and os.path.exists(img_path):
            try:
                ir = ImageReader(img_path)
                c.drawImage(ir, 2*cm, y - 4*cm, width=6*cm, height=3.5*cm, preserveAspectRatio=True)
            except Exception:
                pass
        y -= 5*cm
        if y < 3*cm:
            c.showPage()
            y = ph - 2*cm

    c.save()
    with open(report_path, "rb") as fp:
        st.download_button("Download PDF", fp.read(),
                           file_name=os.path.basename(report_path),
                           mime="application/pdf")
    st.success(f"Report saved to {report_path}")

# Auto-refresh every 4 s so new incidents appear without manual action
time.sleep(4)
st.rerun()
