import os
import io
import json
import requests
import streamlit as st
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# App config
st.set_page_config(
    page_title="BlazeVeritas AI — Wildfire Agent",
    layout="wide",
    page_icon="🔥",
)

# Global CSS
st.markdown("""
<style>
/* hide Streamlit default header/footer */
header[data-testid="stHeader"] {visibility: hidden;}
footer {visibility: hidden;}
/* page padding */
section.main > div {padding-top: 0.6rem;}
/* chips */
.badge {display:inline-block;padding:.25rem .6rem;border-radius:999px;
        font-size:0.8rem;font-weight:600;margin-right:.35rem;}
.badge-ok   {background:#E8FFF2;color:#0B7A43;border:1px solid #B9F2D0;}
.badge-warn {background:#FFF7E6;color:#A15A00;border:1px solid #FFE0A3;}
.badge-err  {background:#FFEAEA;color:#9B1C1C;border:1px solid #FFC2C2;}
/* tighten tabs */
.stTabs [data-baseweb="tab-list"] {gap: .5rem;}
/* nav bar */
.nav {background: linear-gradient(90deg,#0b2a5a,#134a9a); color:#fff;
      border-radius:10px; padding:10px 14px; margin:2px 0 8px 0;}
.nav .row {display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap;}
.nav .left {display:flex; align-items:center; gap:12px;}
.nav .brand {display:flex; align-items:center; gap:10px; font-weight:800; font-size:20px;}
.nav .menu {display:flex; gap:22px; margin-left:18px;}
.nav a {color:#fff; text-decoration:none; font-weight:700;}
.nav a:hover {text-decoration:underline;}
.nav .right {display:flex; align-items:center; gap:12px;}
.nav .pill {background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.18);
            padding:6px 10px; border-radius:16px; font-size:12px;}
.block-container {padding-top:0.6rem;}
</style>
""", unsafe_allow_html=True)


# State & helpers
API_URL_DEFAULT = os.environ.get("API_URL", "http://127.0.0.1:8000")

def get_api_url() -> str:
    """Current backend URL; avoids double-setting warnings."""
    return st.session_state.get("api_url", API_URL_DEFAULT).rstrip("/")

def post_json(path: str, payload: dict):
    try:
        return requests.post(f"{get_api_url()}{path}", json=payload, timeout=120)
    except Exception as e:
        st.error(f"Request error: {e}")
        return None

def post_file(path: str, img: Image.Image):
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG"); buf.seek(0)
        files = {"file": ("upload.png", buf.getvalue(), "image/png")}
        return requests.post(f"{get_api_url()}{path}", files=files, timeout=120)
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None

def absolutize(u: str) -> str:
    """Turn relative 'reports/...' into 'http://.../reports/...'."""
    if not u:
        return u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return f"{get_api_url().rstrip('/')}/{u.lstrip('/')}"

def ping_backend() -> tuple[bool, str]:
    try:
        r = requests.get(f"{get_api_url()}/health", timeout=3)
        if r.ok:
            return True, "Connected"
        return False, f"HTTP {r.status_code}"
    except Exception:
        return False, "Not reachable"

def openai_status() -> tuple[bool, str]:
    on = bool(os.environ.get("OPENAI_API_KEY"))
    return on, "OPENAI key: set" if on else "OPENAI key: missing"

def parse_list(txt, cast=float):
    return [cast(x.strip()) for x in txt.split(",") if x.strip()]

def ece_uniform(probs, labels, n_bins=10):
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="uniform")
    ece = float(np.abs(frac_pos - mean_pred).mean())
    points = [{"prob_bin_center": float(mp), "accuracy": float(fp)} for mp, fp in zip(mean_pred, frac_pos)]
    return ece, points

# Top Nav Bar (WHO-style)

LOGO_SVG = """
<svg width="28" height="28" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#ff6a00"/>
      <stop offset="100%" stop-color="#ff2d55"/>
    </linearGradient>
    <linearGradient id="g2" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0ea5e9"/>
      <stop offset="100%" stop-color="#22c55e"/>
    </linearGradient>
  </defs>
  <!-- shield -->
  <path d="M32 4l18 6v15c0 12-8 21-18 25C22 46 14 37 14 25V10l18-6z"
        fill="url(#g1)"/>
  <!-- flame -->
  <path d="M33 18c4 6-3 7-1 12 2 4 9 3 11-2 3-7-5-11-5-15-3 2-5 3-5 5z"
        fill="#fff" fill-opacity=".9"/>
  <!-- check -->
  <path d="M23 36l5 5 13-13" stroke="url(#g2)" stroke-width="4" fill="none"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

backend_ok, backend_msg = ping_backend()
openai_ok, openai_msg   = openai_status()

backend_badge = f'<span class="badge {"badge-ok" if backend_ok else "badge-err"}">Backend • {backend_msg}</span>'
openai_badge  = f'<span class="badge {"badge-ok" if openai_ok else "badge-warn"}">{openai_msg}</span>'
rag_badge     = '<span class="badge badge-ok">RAG • Ready</span>'

st.markdown(f"""
<div class="nav">
  <div class="row">
    <div class="left">
      <div class="brand">{LOGO_SVG} <span>BlazeVeritas AI</span></div>
      <div class="menu">
        <a href="#home">Home ▾</a>
        <a href="#detect">Wildfire Topics ▾</a>
        <a href="#calib">Countries ▾</a>
        <a href="#copilot">Emergencies ▾</a>
        <a href="#map">About WHO ▾</a>
        <a href="https://github.com/dilrabonu" target="_blank">Docs ▾</a>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)



st.markdown("<a id='home'></a>", unsafe_allow_html=True)


# Tabs

tabs = st.tabs(["🏠 Overview", "🧪 Detect", "📏 Calibration", "🧭 Copilot", "🗺️ Map"])

# Overview

with tabs[0]:
    # Welcome
    st.markdown("""
    <style>
    .hero {
        display: grid;
        grid-template-columns: 1.1fr 1.4fr;
        gap: 20px;
        align-items: center;
        background: linear-gradient(90deg, #fdf7f2 0%, #f1f7ff 100%);
        border: 1px solid #eaeaea;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 1.0rem;
        box-shadow: 0 1px 10px rgba(0,0,0,0.03);
    }
    .hero h2 { margin: 0 0 .25rem 0; font-size: 1.6rem; }
    .hero p  { margin: .25rem 0; font-size: .98rem; }
    .hero small{opacity:.7;}
    .callout {
        background: #eaf3ff;
        color: #0b3a79;
        padding: .9rem 1.0rem;
        border-radius: 12px;
        border: 1px solid #dbe9ff;
        margin-top: .75rem;
        font-weight: 600;
    }
    .shadow-img { border-radius: 12px; overflow:hidden; box-shadow: 0 5px 18px rgba(0,0,0,.08); }
    </style>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown('<div class="shadow-img">', unsafe_allow_html=True)
        
        st.image(
            "https://images.unsplash.com/photo-1473773508845-188df298d2d1?q=80&w=1600&auto=format&fit=crop",
            use_column_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("### Welcome to BlazeVeritas AI")
        st.markdown(
            "Your **wildfire detection & decision copilot**. "
            "Upload imagery, get an explainable prediction with **Grad-CAM**, "
            "and generate a cited action plan with our **OpenAI + LangChain RAG** pipeline."
        )
        st.markdown(
            "<small>Built for speed, clarity, and real-world incident response.</small>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="callout">🚀 Faster, explainable triage reduces false dispatches and time-to-decision.</div>',
                    unsafe_allow_html=True)

    # WHAT IS THIS? 
    st.markdown("## What is this?")
    st.markdown("""
- **Detects wildfire cues** from imagery (ResNet-18).
- **Explains** each prediction with **Grad-CAM**.
- **Calibrates** probabilities (temperature scaling) + **MC-Dropout** uncertainty.
- **Advises** responders with an **OpenAI + LangChain RAG Copilot** grounded in your docs.
    """)

    st.info("Early, explainable detection + cited action plans reduce false dispatches and time-to-decision.")
# Detect
with tabs[1]:
    st.markdown("<a id='detect'></a>", unsafe_allow_html=True)
    st.subheader("Detect (with Grad-CAM)")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Upload Image")
        up = st.file_uploader("JPG/PNG", type=["jpg", "jpeg", "png", "jfif"])
        if up is not None:
            img = Image.open(up).convert("RGB")
            st.image(img, caption="Input", use_column_width=True)
            show_raw = st.toggle("Show raw input next to Grad-CAM", value=True)
            if st.button("Run Detection (Upload)", use_container_width=True):
                with st.spinner("Calling API..."):
                    r = post_file("/v1/detect", img)
                if r and r.ok:
                    d = r.json()
                    st.success(f"**{d['label'].upper()}** | p={d['prob']:.3f} | H={d['uncertainty']:.3f} • event_id: {d['event_id']}")
                    gc = absolutize(d.get("grad_cam_url", ""))
                    if show_raw:
                        a, b = st.columns(2)
                        with a: st.image(gc, caption="Grad-CAM overlay", use_column_width=True)
                        with b: st.image(img, caption="Original", use_column_width=True)
                    else:
                        st.image(gc, caption="Grad-CAM overlay", use_column_width=True)
                elif r:
                    st.error(r.text)

    
    with st.expander("🛠 If results never change (debug checklist)"):
        st.markdown("""
- **Weights loaded?** Backend log should say `[weights] loaded=... | missing=0`  
- **Normalization mismatch?** Try `TEMP_INIT=1.0` in `api/settings.py`  
- **Grad-CAM files?** Images saved under `reports/xai/`  
- **Varied inputs?** Test clean sky vs smoke vs fire
""")

# Calibration
with tabs[2]:
    st.markdown("<a id='calib'></a>", unsafe_allow_html=True)
    st.header("Calibration & Uncertainty")
    st.caption("Paste predicted **probs** and **labels** to compute **ECE** and visualize the reliability curve.")

    c1, c2 = st.columns(2)
    with c1:
        probs_txt = st.text_area("probs (comma-separated)", "0.1,0.6,0.8,0.2,0.95", key="calib_probs")
    with c2:
        labels_txt = st.text_area("labels (0/1, comma-separated)", "0,1,1,0,1", key="calib_labels")

    n_bins = st.slider("Number of bins", 2, 20, 10, 1)

    if st.button("Compute ECE", type="primary", key="compute_ece"):
        try:
            p = parse_list(probs_txt, float)
            y = parse_list(labels_txt, int)
            if len(p) != len(y):
                st.error(f"Length mismatch: {len(p)} probs vs {len(y)} labels.")
            elif len(p) < 3:
                st.warning("Give at least 3–5 samples for a meaningful curve.")
            else:
                ece, points = ece_uniform(np.array(p), np.array(y), n_bins=n_bins)
                st.success(f"ECE: **{ece:.4f}**")

                # Show raw points
                st.json(points)

                # Plot reliability diagram
                try:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(6, 4))
                    xs = [pt["prob_bin_center"] for pt in points]
                    ys = [pt["accuracy"] for pt in points]
                    plt.plot([0, 1], [0, 1], "--", linewidth=1)
                    plt.plot(xs, ys, marker="o", linewidth=2)
                    plt.xlabel("Mean predicted probability")
                    plt.ylabel("Empirical accuracy")
                    plt.title("Reliability Diagram")
                    st.pyplot(fig, clear_figure=True)

                    # Histogram of probabilities
                    fig2 = plt.figure(figsize=(6, 2.8))
                    plt.hist(p, bins=n_bins, alpha=0.85)
                    plt.xlabel("Predicted probability")
                    plt.ylabel("Count")
                    plt.title("Probability Histogram")
                    st.pyplot(fig2, clear_figure=True)
                except ModuleNotFoundError:
                    st.info("Install matplotlib for charts: `pip install matplotlib`")
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[3]:

    st.markdown("""
    <style>
      .copilot-wrap {max-width: 1200px; margin: 0 auto;}
      .cp-title {font-size: 1.6rem; font-weight: 800; margin-bottom: .25rem;}
      .cp-sub {opacity:.75; margin-bottom: 1.2rem;}
      .cp-chiprow {display:flex; gap:10px; flex-wrap:wrap; margin:.3rem 0 1rem;}
      .cp-chip {background:#f1f5f9; border:1px solid #e5e7eb; padding:8px 12px; border-radius:999px; font-weight:600; cursor:pointer;}
      .cp-chip:hover {background:#eef2ff;}
      .cp-card {border:1px solid #e9eaef; background:#fff; border-radius:16px; padding:16px;}
      .cp-hint {opacity:.60;}
    </style>
    """, unsafe_allow_html=True)

    # minimal helper
    def _send_to_copilot(objective: str):
        """Minimal call to your FastAPI /v1/copilot/plan endpoint with defaults."""
        payload = dict(
            # No event context – keep it minimal; backend can handle None/defaults
            lat=None, lon=None, label="unknown", prob=0.0, uncertainty=0.0,
            objectives=objective,
            _ui={"model": "gpt-4o-mini"}   # optional UI hint (not shown in UI)
        )
        return post_json("/v1/copilot/plan", payload)

    # session history
    if "copilot_msgs" not in st.session_state:
        st.session_state.copilot_msgs = []  # list of {"role": "user"|"assistant", "text": str}

    st.markdown("<div class='copilot-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='cp-title'>Hello! 🌤 — What wildfire insight can I help with today?</div>", unsafe_allow_html=True)
    st.markdown("<div class='cp-sub'>RAG-grounded answers and structured action plans — in a clean, chat-first UI.</div>", unsafe_allow_html=True)

     
    colS = st.container()
    with colS:
        st.markdown("<div class='cp-chiprow'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("🛰 Analyze image", use_container_width=True):
                prompt = ("Analyze the latest detection image if available. "
                          "Explain visual cues of fire/smoke, and list validation steps.")
                r = _send_to_copilot(prompt)
                if r and r.ok:
                    st.session_state.copilot_msgs += [
                        {"role": "user", "text": prompt},
                        {"role": "assistant", "text": r.json().get("plan","")}
                    ]
                elif r:
                    st.error(r.text)

        with c2:
            if st.button("📊 Summarize report", use_container_width=True):
                prompt = ("Summarize key risks and mitigation strategies for the current region based on retrieved docs. "
                          "Use concise bullets and cite sources if available.")
                r = _send_to_copilot(prompt)
                if r and r.ok:
                    st.session_state.copilot_msgs += [
                        {"role": "user", "text": prompt},
                        {"role": "assistant", "text": r.json().get("plan","")}
                    ]
                elif r:
                    st.error(r.text)

        with c3:
            if st.button("🧭 Generate plan", use_container_width=True):
                prompt = ("Generate an action plan with: Situation summary, confidence & uncertainties, key risks "
                          "(wind/fuel/slope), Immediate Actions (0–30 min), Next Steps (2–6 hr), "
                          "Public comms template, and cited sources.")
                r = _send_to_copilot(prompt)
                if r and r.ok:
                    st.session_state.copilot_msgs += [
                        {"role": "user", "text": prompt},
                        {"role": "assistant", "text": r.json().get("plan","")}
                    ]
                elif r:
                    st.error(r.text)

        with c4:
            if st.button("💡 Ask weather impact", use_container_width=True):
                prompt = ("How will wind direction and humidity affect potential fire spread in the next 6 hours? "
                          "Include uncertainties and verification steps.")
                r = _send_to_copilot(prompt)
                if r and r.ok:
                    st.session_state.copilot_msgs += [
                        {"role": "user", "text": prompt},
                        {"role": "assistant", "text": r.json().get("plan","")}
                    ]
                elif r:
                    st.error(r.text)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat area 
    chat_box = st.container()
    with chat_box:
        st.markdown("### Copilot")
        with st.container(border=True):
            if not st.session_state.copilot_msgs:
                st.markdown("<div class='cp-hint'>No messages yet. Use a quick action above, or ask a question below.</div>", unsafe_allow_html=True)
            else:
                for m in st.session_state.copilot_msgs:
                    if m["role"] == "user":
                        st.markdown(f"**You:** {m['text']}")
                    else:
                        st.markdown(m["text"])

        # input row
        q = st.text_input(
            "Ask the Copilot",
            key="cp_input_min",
            placeholder="Ask anything… e.g., “What are the immediate risks near the detection area?”",
            label_visibility="collapsed"
        )
        send = st.button("Send", type="primary")
        if send and q.strip():
            with st.spinner("Thinking with RAG…"):
                r = _send_to_copilot(q.strip())
            if r and r.ok:
                ans = r.json().get("plan", "")
                st.session_state.copilot_msgs += [
                    {"role": "user", "text": q.strip()},
                    {"role": "assistant", "text": ans},
                ]
                st.rerun()
            elif r:
                st.error(r.text)

    st.markdown("</div>", unsafe_allow_html=True)
# Map

with tabs[4]:
    st.markdown("<a id='map'></a>", unsafe_allow_html=True)
    st.subheader("Map & Layers (Coming Soon)")
    st.info("This tab will overlay detections, FIRMS, wind vectors, and buffers.")
