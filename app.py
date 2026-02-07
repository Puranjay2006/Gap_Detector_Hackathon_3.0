"""
FutureFormers: Route Continuity Gap Detector
=============================================
AI-powered QA tool that detects ONE specific error type in road network data:
**Endpoint Gaps** — where road segments should connect but have small coordinate
gaps between their endpoints, breaking route continuity.

Uses Feature Extraction + Rule-Based Gap Detection + Isolation Forest ML
to find, visualize, and auto-fix broken connections.

Team: FutureFormers | IIT Mandi Hackathon 3.0 | Problem Statement 2
Members: Puranjay Gambhir, Akshobhya Rao, Rohan
Company: Axes Systems GmbH | Option B: Rule-Based Geometry Validation
Version: 7.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import json
from datetime import datetime
from streamlit_folium import st_folium
from shapely import wkt
from shapely.geometry import Point, LineString
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_CONFIG = {
    "title": "Route Continuity Gap Detector",
    "version": "7.0.0",
    "icon": "🔍",
    "precision": 6,
    "team": "FutureFormers",
    "members": ["Puranjay Gambhir", "Akshobhya Rao", "Rohan"],
    "error_type": "ENDPOINT_GAP",
    "error_label": "Route Continuity Gap",
}

# =============================================================================
# DEMO DATA — Problem 2 streets_xgen.wkt (56 LINESTRINGs)
# =============================================================================
DEMO_WKT_DATA = """LINESTRING(7071.421606 8585.627528, 7074.672945 8588.813669, 7074.902551 8589.026835, 7075.084535 8589.196346, 7079.265638 8592.884220, 7081.745726 8594.906230, 7087.160750 8598.968277, 7091.114457 8601.621165, 7097.122261 8605.189984, 7104.911244 8609.486740, 7105.304126 8609.703307, 7105.767874 8609.959559, 7144.536643 8631.196101, 7154.035087 8636.400567, 7154.349676 8636.571269, 7154.424397 8636.611861, 7163.899654 8641.749543, 7194.100989 8658.211975, 7204.788850 8664.037795, 7217.453480 8671.075087, 7225.937575 8676.277228, 7229.984882 8679.124346, 7234.367017 8682.488447, 7238.456504 8685.821480, 7243.451150 8690.254299, 7247.565354 8694.303307, 7251.962457 8699.059843, 7255.123087 8702.735811, 7259.170961 8707.866520, 7262.826576 8713.000630, 7262.874709 8713.070759, 7263.406488 8713.854425, 7268.192504 8721.195591, 7269.545140 8723.350488)
LINESTRING(7228.943036 8691.704220, 7230.263868 8692.301820)
LINESTRING(7230.130016 8690.909669, 7236.635528 8693.947843, 7238.203087 8694.558992, 7246.566879 8698.129852)
LINESTRING(7233.052082 8685.026022, 7233.114331 8684.900731)
LINESTRING(7235.343383 8681.162627, 7235.951754 8680.340183)
LINESTRING(7233.114331 8684.900731, 7234.209751 8682.695150, 7234.367017 8682.488447, 7234.542142 8682.245745, 7235.343383 8681.162627)
LINESTRING(7230.130016 8690.909669, 7233.052082 8685.026022)
LINESTRING(7255.510243 8699.099074, 7250.881096 8693.964737, 7245.560806 8688.592006, 7239.691219 8683.390375, 7235.951754 8680.340183)
LINESTRING(7219.235906 8686.145197, 7230.130016 8690.909669)
LINESTRING(7171.129701 8650.784296, 7172.734677 8650.679301, 7173.495099 8650.929033, 7174.214079 8651.165216, 7192.126998 8660.929323, 7197.898450 8664.075326)
LINESTRING(7150.309569 8645.723660, 7151.809323 8646.372283, 7152.455452 8646.687836)
LINESTRING(7178.488951 8636.167899, 7179.095849 8636.294381)
LINESTRING(7235.951754 8680.340183, 7235.062809 8679.661965, 7231.893165 8677.230009, 7226.680365 8673.607106, 7220.970935 8670.106261, 7215.024189 8666.676907, 7210.789172 8664.323641, 7203.506230 8660.303943, 7191.076082 8653.528460, 7187.602110 8651.634803, 7183.533657 8649.417090, 7179.077877 8646.988365, 7173.680598 8644.046343)
LINESTRING(7173.680598 8644.046343, 7178.431691 8637.105203, 7179.218646 8635.065449)
LINESTRING(7152.922545 8639.584441, 7153.182312 8638.969380)
LINESTRING(7152.922545 8639.584441, 7153.400580 8639.845512, 7155.481039 8640.973587, 7159.220787 8643.001380, 7165.913783 8646.640781, 7168.378110 8647.984063, 7170.192680 8648.973184, 7170.323414 8649.225921, 7170.529550 8649.624302, 7171.129701 8650.784296)
LINESTRING(7149.314835 8641.775055, 7151.010860 8642.887597, 7150.309569 8645.723660)
LINESTRING(7159.331735 8627.200894, 7160.009443 8627.429027, 7160.622293 8627.964718)
LINESTRING(7160.622293 8627.964718, 7161.134457 8628.255496, 7161.850828 8628.826280, 7163.425814 8629.276309, 7164.344580 8629.632567, 7165.696139 8630.103005, 7169.024466 8631.261468, 7171.403754 8632.280239, 7173.416693 8633.310123, 7174.659005 8634.026778, 7175.660031 8634.604195, 7177.199414 8635.492233, 7178.007798 8635.958532, 7178.488951 8636.167899)
LINESTRING(7160.535723 8626.610494, 7160.009443 8627.429027, 7159.280315 8628.578646, 7155.343729 8634.079106)
LINESTRING(7155.846879 8634.352252, 7154.349676 8636.571269, 7153.182312 8638.969380)
LINESTRING(7155.343729 8634.079106, 7156.744044 8634.838564, 7157.181770 8635.075880, 7158.321865 8635.694060, 7164.473783 8639.029814, 7171.316220 8642.757487, 7173.680598 8644.046343)
LINESTRING(7155.846879 8634.352252, 7155.343729 8634.079106)
LINESTRING(7160.535723 8626.610494, 7163.093197 8627.735509, 7169.459698 8630.137304, 7172.826746 8631.579005, 7178.738627 8634.989310, 7179.218646 8635.065449)
LINESTRING(7152.922545 8639.584441, 7146.890306 8636.283666, 7136.488517 8630.585235, 7134.319162 8629.396894, 7124.383899 8624.969065, 7123.323969 8624.777953, 7121.539899 8624.753065, 7118.794885 8624.141065, 7118.036787 8624.066457, 7117.057928 8624.132050, 7116.004913 8624.447093)
LINESTRING(7116.004913 8624.447093, 7112.647899 8624.312050, 7111.216913 8624.249065, 7107.446608 8624.105802, 7101.497594 8623.934816, 7098.702406 8624.024447)
LINESTRING(7097.387528 8625.209953, 7096.202759 8628.525808, 7096.091868 8628.836088, 7094.616094 8632.063616, 7094.575502 8632.152454, 7094.528787 8632.254614, 7094.032838 8633.339263, 7097.047370 8635.481575)
LINESTRING(7098.702406 8624.024447, 7099.318488 8623.211528)
LINESTRING(7097.387528 8625.209953, 7098.702406 8624.024447)
LINESTRING(7089.693165 8635.350614, 7093.515969 8630.246551, 7097.387528 8625.209953)
LINESTRING(7104.226620 8612.107370, 7104.311093 8611.937461)
LINESTRING(7099.318488 8623.211528, 7104.226620 8612.107370)
LINESTRING(7097.887446 8624.084598, 7083.675950 8619.934734, 7065.840416 8611.314803)
LINESTRING(7065.840416 8611.314803, 7065.689216 8610.036491, 7065.330009 8608.768101)
LINESTRING(7065.849090 8609.703591, 7066.134992 8610.118299, 7083.707584 8618.672806, 7085.072580 8619.137802, 7099.318488 8623.211528)
LINESTRING(7065.849090 8609.703591, 7065.689216 8610.036491, 7065.592441 8610.237921)
LINESTRING(7115.152479 8607.998154, 7118.572479 8608.358154, 7119.388517 8608.574154, 7124.440479 8611.886154, 7128.544535 8614.742117, 7129.348498 8615.426117, 7136.590847 8621.963206, 7137.610186 8622.521575, 7138.110047 8622.795345, 7155.354331 8626.387465, 7159.331735 8627.200894)
LINESTRING(7155.343729 8634.079106, 7154.315490 8633.515861, 7153.494236 8633.065890, 7144.602803 8628.194211, 7134.657279 8622.746306, 7128.239131 8619.230608, 7122.178772 8615.910898, 7116.403011 8612.747093, 7112.053701 8610.364630, 7108.643225 8608.496485, 7106.827238 8606.270154)
LINESTRING(7106.625298 8606.725342, 7106.827238 8606.270154)
LINESTRING(7106.625298 8606.725342, 7105.304126 8609.703307, 7104.311093 8611.937461)
LINESTRING(7106.827238 8606.270154, 7109.809455 8599.548132, 7110.782362 8597.356157)
LINESTRING(7079.423924 8595.277739, 7078.770085 8595.916724, 7076.599654 8596.376220)
LINESTRING(7076.599654 8596.376220, 7073.598444 8593.672479)
LINESTRING(7073.598444 8593.672479, 7072.557335 8594.218658, 7071.476598 8594.855376)
LINESTRING(7102.205405 8610.012227, 7095.419206 8606.216466, 7089.858992 8602.878217, 7084.575269 8599.029789, 7079.423924 8595.277739)
LINESTRING(7074.076422 8591.860120, 7073.526784 8591.375282)
LINESTRING(7072.623950 8594.066154, 7072.557335 8594.218658, 7065.849090 8609.703591)
LINESTRING(7073.526784 8591.375282, 7074.141449 8590.748598, 7074.097682 8590.508277)
LINESTRING(7104.226620 8612.107370, 7097.580850 8608.486054, 7094.466028 8606.711339, 7090.380907 8604.284655, 7087.550343 8602.438507, 7083.294236 8599.398803, 7081.232315 8597.852050, 7078.770085 8595.916724, 7074.076422 8591.860120)
LINESTRING(7072.623950 8594.066154, 7073.270646 8592.708359, 7074.076422 8591.860120)
LINESTRING(7076.039187 8586.425254, 7076.445109 8585.496170)
LINESTRING(7073.644989 8591.195792, 7074.097682 8590.508277, 7074.902551 8589.026835, 7076.039187 8586.425254)
LINESTRING(7076.445109 8585.496170, 7078.460542 8580.883238)
LINESTRING(7078.460542 8580.883238, 7088.449323 8586.079937, 7107.773102 8595.942917, 7107.972548 8596.047402, 7110.782362 8597.356157)"""

# =============================================================================
# ERROR EXAMPLES — for training & demonstration
# =============================================================================

EXAMPLE_CORRECT = """LINESTRING(100 200, 120 210, 140 220, 160 230)
LINESTRING(160 230, 180 240, 200 250, 220 260)
LINESTRING(220 260, 240 270, 260 280, 280 290)
LINESTRING(160 230, 170 215, 180 200, 190 185)
LINESTRING(220 260, 230 275, 240 290, 250 305)"""

EXAMPLE_ERROR_1 = """LINESTRING(100 200, 120 210, 140 220, 160 230)
LINESTRING(160.5 230.3, 180 240, 200 250, 220 260)
LINESTRING(220 260, 240 270, 260 280, 280 290)
LINESTRING(160 230, 170 215, 180 200, 190 185)
LINESTRING(220.8 260.5, 230 275, 240 290, 250 305)"""

EXAMPLE_ERROR_2 = """LINESTRING(500 600, 520 610, 540 620, 560 630)
LINESTRING(560 630, 580 640, 600 650, 620 660)
LINESTRING(620.2 660.1, 640 670, 660 680, 680 690)
LINESTRING(560.4 630.2, 570 615, 580 600, 590 585)
LINESTRING(680 690, 700 700, 720 710)
LINESTRING(720.3 710.2, 740 720, 760 730)"""

EXAMPLE_ERROR_3 = """LINESTRING(300 400, 320 410, 340 420, 360 430)
LINESTRING(360.1 430.05, 380 440, 400 450, 420 460)
LINESTRING(420 460, 440 470, 460 480)
LINESTRING(460 480, 480 490, 500 500)
LINESTRING(500.6 500.4, 520 510, 540 520)
LINESTRING(360 430, 365 415, 370 400)
LINESTRING(370.3 400.15, 375 385, 380 370)"""


# =============================================================================
# CUSTOM CSS — GLASSMORPHISM + DM SANS
# =============================================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

    :root {
        --primary: #6366f1; --primary-dark: #4f46e5; --primary-light: #818cf8;
        --secondary: #06b6d4; --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
        --dark: #0f172a; --gray-900: #1e293b; --gray-800: #334155; --gray-600: #475569;
        --gray-400: #94a3b8; --gray-200: #e2e8f0; --gray-100: #f1f5f9; --white: #ffffff;
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    }

    *, html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }

    .main .block-container { padding: 2rem 2rem 3rem 2rem; max-width: 1400px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.97) 0%, rgba(30,41,59,0.97) 100%);
        backdrop-filter: blur(20px); border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] * { color: var(--gray-100) !important; }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: var(--white) !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; margin: 1.5rem 0; }

    .hero-container {
        background: rgba(255,255,255,0.92); backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.5); border-radius: 32px;
        padding: 2.5rem 2rem; margin-bottom: 2rem; text-align: center;
        box-shadow: var(--shadow-xl); position: relative; overflow: hidden;
    }
    .hero-container::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary), var(--primary-light));
        background-size: 200% 100%; animation: grad 3s ease infinite;
    }
    @keyframes grad { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
    .hero-icon { font-size: 3.5rem; margin-bottom: 0.25rem; animation: float 3s ease-in-out infinite; }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
    .hero-title {
        font-size: 2.75rem; font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark), #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 0 0 0.4rem 0; letter-spacing: -0.02em;
    }
    .hero-subtitle { font-size: 1.15rem; color: var(--gray-600); font-weight: 500; margin: 0; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.4rem 1rem; background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white !important; border-radius: 100px; font-size: 0.8rem; font-weight: 600;
        margin-top: 0.8rem; box-shadow: 0 4px 14px 0 rgba(99,102,241,0.4);
    }

    .metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
    @media(max-width:1024px){ .metric-grid{grid-template-columns:repeat(2,1fr)} }
    .metric-card {
        background: rgba(255,255,255,0.92); backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.5); border-radius: 20px;
        padding: 1.25rem; text-align: center; box-shadow: var(--shadow-lg);
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1); position: relative; overflow: hidden;
    }
    .metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:20px 20px 0 0; }
    .metric-card.purple::before { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
    .metric-card.blue::before { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
    .metric-card.red::before { background: linear-gradient(90deg,#ef4444,#f87171); }
    .metric-card.green::before { background: linear-gradient(90deg,#10b981,#34d399); }
    .metric-card:hover { transform: translateY(-5px) scale(1.02); box-shadow: 0 20px 40px -12px rgba(0,0,0,0.2); }
    .metric-card { animation: fadeInUp 0.5s ease-out both; }
    .metric-card:nth-child(1) { animation-delay: 0.05s; }
    .metric-card:nth-child(2) { animation-delay: 0.1s; }
    .metric-card:nth-child(3) { animation-delay: 0.15s; }
    .metric-card:nth-child(4) { animation-delay: 0.2s; }
    .metric-icon { font-size: 1.75rem; margin-bottom: 0.3rem; }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: var(--gray-900); line-height: 1; margin-bottom: 0.15rem; }
    .metric-label { font-size: 0.8rem; font-weight: 600; color: var(--gray-600); text-transform: uppercase; letter-spacing: 0.05em; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem; background: rgba(255,255,255,0.6); backdrop-filter: blur(12px);
        border-radius: 16px; padding: 0.4rem; border: 1px solid rgba(255,255,255,0.3);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 12px; padding: 0.65rem 1.25rem;
        font-weight: 600; color: var(--gray-600); transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(99,102,241,0.1); color: var(--primary); transform: translateY(-1px); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
        color: white !important; box-shadow: 0 4px 14px 0 rgba(99,102,241,0.4);
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255,255,255,0.87); backdrop-filter: blur(16px);
        border-radius: 24px; padding: 1.75rem; margin-top: 0.75rem;
        border: 1px solid rgba(255,255,255,0.4); box-shadow: var(--shadow-lg);
    }

    .section-header { display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem; }
    .section-header h3 { font-size:1.4rem; font-weight:700; color:var(--gray-900); margin:0; }
    .section-header .icon { font-size:1.5rem; }

    .map-container { border-radius:20px; overflow:hidden; box-shadow:var(--shadow-lg); border:3px solid rgba(255,255,255,0.5); }
    .stDataFrame { border-radius:16px; overflow-x:auto; box-shadow:0 4px 6px -1px rgb(0 0 0/0.1); }
    .stDataFrame [data-testid="stDataFrameResizable"] { min-width:800px; }

    /* Smooth fade-in animation for all content */
    @keyframes fadeInUp { 0% { opacity:0; transform:translateY(20px); } 100% { opacity:1; transform:translateY(0); } }
    @keyframes fadeIn { 0% { opacity:0; } 100% { opacity:1; } }
    @keyframes scaleIn { 0% { opacity:0; transform:scale(0.95); } 100% { opacity:1; transform:scale(1); } }
    @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }

    .hero-container { animation: fadeInUp 0.6s ease-out; }
    .metric-grid { animation: fadeInUp 0.6s ease-out 0.1s both; }
    .stTabs { animation: fadeInUp 0.6s ease-out 0.2s both; }
    .welcome-card { animation: scaleIn 0.5s ease-out; }
    .how-it-works { animation: fadeInUp 0.7s ease-out 0.15s both; }
    .legend-box { animation: fadeIn 0.5s ease-out 0.3s both; }
    .onboarding-overlay { animation: scaleIn 0.4s ease-out; }
    .example-card { animation: fadeInUp 0.4s ease-out; }
    .quality-score { animation: scaleIn 0.5s ease-out; }

    .stButton button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white; border: none; border-radius: 12px; padding: 0.7rem 1.4rem;
        font-weight: 600; font-size: 0.9rem;
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
        box-shadow: 0 4px 14px 0 rgba(99,102,241,0.4);
        white-space: nowrap; position: relative; overflow: hidden;
    }
    .stButton button::after {
        content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    .stButton button:hover::after { left: 100%; }
    .stButton button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px 0 rgba(99,102,241,0.5);
    }
    .stButton button:active { transform: translateY(-1px) scale(0.98); transition: all 0.1s; }
    .stDownloadButton button {
        background: linear-gradient(135deg, #10b981, #059669);
        box-shadow: 0 4px 14px 0 rgba(16,185,129,0.4);
    }
    .stDownloadButton button:hover {
        box-shadow: 0 8px 25px 0 rgba(16,185,129,0.5);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.1); border-radius: 12px; padding: 0.5rem;
        border: 2px dashed rgba(255,255,255,0.3);
    }
    /* Fix file uploader inner area text — dark text on light dropzone */
    [data-testid="stFileUploader"] section { background: rgba(255,255,255,0.95) !important; border-radius: 10px !important; }
    [data-testid="stFileUploader"] section * { color: var(--gray-600) !important; }
    [data-testid="stFileUploader"] section button { color: var(--primary) !important; border-color: var(--primary) !important; }
    [data-testid="stFileUploader"] section small { color: var(--gray-400) !important; }
    [data-testid="stFileUploader"] > label p { color: var(--gray-100) !important; }

    .stAlert { border-radius:16px; border:none; backdrop-filter:blur(8px); }
    .stAlert > div { border-radius:16px; }

    .welcome-card {
        background: rgba(255,255,255,0.92); backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.5); border-radius: 28px;
        padding: 3rem; text-align: center; box-shadow: var(--shadow-xl);
    }
    .welcome-icon { font-size:5rem; margin-bottom:1rem; animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.05)} }
    .welcome-title { font-size:2rem; font-weight:700; color:var(--gray-900); margin-bottom:1rem; }
    .welcome-text { font-size:1.05rem; color:var(--gray-600); line-height:1.7; max-width:600px; margin:0 auto 2rem auto; }
    .step-list { display:flex; flex-direction:column; gap:0.85rem; max-width:480px; margin:0 auto; text-align:left; }
    .step-item {
        display:flex; align-items:center; gap:1rem; padding:0.9rem 1.1rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08));
        border-radius:14px; transition:all 0.2s ease;
    }
    .step-item:hover { transform:translateX(8px); background:linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15)); }
    .step-item { animation: fadeInUp 0.4s ease-out both; }
    .step-item:nth-child(1) { animation-delay: 0.1s; }
    .step-item:nth-child(2) { animation-delay: 0.2s; }
    .step-item:nth-child(3) { animation-delay: 0.3s; }
    .step-item:nth-child(4) { animation-delay: 0.4s; }
    .step-number {
        width:30px; height:30px; background:linear-gradient(135deg, var(--primary), var(--primary-dark));
        color:white; border-radius:10px; display:flex; align-items:center; justify-content:center;
        font-weight:700; font-size:0.8rem; flex-shrink:0;
    }
    .step-text { font-size:0.95rem; font-weight:500; color:var(--gray-800); }

    .stats-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:1.25rem; }
    .stat-card {
        background:linear-gradient(135deg, rgba(99,102,241,0.05), rgba(139,92,246,0.05));
        border-radius:16px; padding:1.25rem; border:1px solid rgba(99,102,241,0.1);
    }
    .stat-card h4 { font-size:0.95rem; font-weight:700; color:var(--primary); margin:0 0 0.75rem 0; display:flex; align-items:center; gap:0.5rem; }
    .stat-item { display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid rgba(99,102,241,0.1); }
    .stat-item:last-child { border-bottom:none; }
    .stat-label { font-size:0.85rem; color:var(--gray-600); }
    .stat-value { font-size:0.95rem; font-weight:700; color:var(--gray-900); }

    .quality-score { background:rgba(255,255,255,0.9); border-radius:20px; padding:1.75rem; text-align:center; margin-top:1.25rem; border:1px solid rgba(255,255,255,0.3); }
    .score-circle {
        width:130px; height:130px; border-radius:50%; display:flex; flex-direction:column;
        align-items:center; justify-content:center; margin:0 auto 0.75rem auto;
        font-weight:800; font-size:2.25rem; color:white; box-shadow:var(--shadow-lg);
    }
    .score-circle.excellent { background:linear-gradient(135deg,#10b981,#059669); }
    .score-circle.good { background:linear-gradient(135deg,#f59e0b,#d97706); }
    .score-circle.poor { background:linear-gradient(135deg,#ef4444,#dc2626); }
    .score-label { font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.4rem; color:var(--gray-600); }

    .example-card {
        background: rgba(255,255,255,0.9); border-radius: 20px; padding: 1.5rem;
        border: 1px solid rgba(99,102,241,0.15); margin-bottom: 1.25rem;
    }
    .example-card.correct { border-left: 4px solid #10b981; }
    .example-card.error { border-left: 4px solid #ef4444; }

    .legend-box {
        background: rgba(255,255,255,0.95); border-radius: 16px; padding: 1.25rem;
        border: 1px solid rgba(99,102,241,0.12); margin-top: 1rem;
    }
    .legend-title { font-size: 0.95rem; font-weight: 700; color: var(--gray-900); margin: 0 0 0.75rem; }
    .legend-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 0.5rem 1.5rem; }
    .legend-item { display: flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0; }
    .legend-swatch { width: 18px; height: 18px; border-radius: 6px; flex-shrink: 0; border: 2px solid rgba(0,0,0,0.08); }
    .legend-label { font-size: 0.82rem; color: var(--gray-600); line-height: 1.3; }
    .legend-label b { color: var(--gray-800); }

    .how-it-works {
        background: rgba(255,255,255,0.92); backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.5); border-radius: 28px;
        padding: 2.5rem; margin-top: 2rem; box-shadow: var(--shadow-xl);
    }
    .hiw-title { font-size: 1.75rem; font-weight: 800; color: var(--gray-900); text-align: center; margin: 0 0 0.4rem; }
    .hiw-subtitle { font-size: 1rem; color: var(--gray-600); text-align: center; margin: 0 0 2rem; }
    .pipeline-steps { display: flex; flex-direction: column; gap: 0; max-width: 700px; margin: 0 auto; }
    .pipe-step {
        display: flex; gap: 1.25rem; align-items: flex-start; position: relative;
        padding: 1.25rem 0; animation: fadeInUp 0.5s ease-out both;
    }
    .pipe-step:nth-child(1) { animation-delay: 0.1s; }
    .pipe-step:nth-child(2) { animation-delay: 0.2s; }
    .pipe-step:nth-child(3) { animation-delay: 0.3s; }
    .pipe-step:nth-child(4) { animation-delay: 0.4s; }
    .pipe-step:nth-child(5) { animation-delay: 0.5s; }
    .pipe-step:not(:last-child)::after {
        content: ''; position: absolute; left: 24px; top: 60px; bottom: 0;
        width: 3px; background: linear-gradient(180deg, var(--primary-light), rgba(99,102,241,0.15));
    }
    .pipe-icon {
        width: 50px; height: 50px; border-radius: 16px; display: flex; align-items: center;
        justify-content: center; font-size: 1.5rem; flex-shrink: 0; color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12); z-index: 1;
    }
    .pipe-icon.a { background: linear-gradient(135deg, #6366f1, #4f46e5); }
    .pipe-icon.b { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .pipe-icon.c { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
    .pipe-icon.d { background: linear-gradient(135deg, #06b6d4, #0891b2); }
    .pipe-icon.e { background: linear-gradient(135deg, #10b981, #059669); }
    .pipe-content { flex: 1; }
    .pipe-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--primary); margin: 0 0 0.15rem; }
    .pipe-title { font-size: 1.1rem; font-weight: 700; color: var(--gray-900); margin: 0 0 0.3rem; }
    .pipe-desc { font-size: 0.88rem; color: var(--gray-600); line-height: 1.5; margin: 0; }

    .onboarding-overlay {
        background: rgba(255,255,255,0.95); backdrop-filter: blur(16px);
        border: 2px solid var(--primary); border-radius: 20px;
        padding: 1.5rem; margin-bottom: 1.5rem; position: relative;
        box-shadow: 0 0 0 4px rgba(99,102,241,0.15), var(--shadow-xl);
        animation: onboard-pulse 2s ease-in-out infinite;
    }
    @keyframes onboard-pulse {
        0%,100% { box-shadow: 0 0 0 4px rgba(99,102,241,0.15), var(--shadow-xl); }
        50% { box-shadow: 0 0 0 8px rgba(99,102,241,0.1), var(--shadow-xl); }
    }
    .onboarding-step-badge {
        position: absolute; top: -12px; left: 20px;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white; padding: 0.2rem 0.9rem; border-radius: 100px;
        font-size: 0.75rem; font-weight: 700;
    }
    .onboarding-title { font-size: 1.15rem; font-weight: 700; color: var(--gray-900); margin: 0.5rem 0 0.3rem; }
    .onboarding-desc { font-size: 0.9rem; color: var(--gray-600); line-height: 1.5; margin: 0; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;}

    /* Hide ALL Material Icon text leaks (keyboard_double_arrow, expand_more, etc.) */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button { font-size: 0 !important; line-height: 0 !important; overflow: hidden !important; }
    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="collapsedControl"] button svg { width: 24px !important; height: 24px !important; font-size: 24px !important; }
    [data-testid="stSidebarCollapseButton"] button span,
    [data-testid="collapsedControl"] button span { font-size: 0 !important; display: none !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] { background: transparent !important; }

    /* Hide Material Icon text leaks — only target icon containers, not label text */
    [data-testid="stExpanderToggleIcon"] { font-size: 0 !important; overflow: hidden !important; width: 24px !important; height: 24px !important; display: inline-flex !important; align-items: center !important; justify-content: center !important; }
    [data-testid="stExpanderToggleIcon"] svg { font-size: 20px !important; width: 20px !important; height: 20px !important; }
    [data-testid="stIconMaterial"] { font-size: 0 !important; overflow: hidden !important; display: inline-flex !important; width: 24px !important; height: 24px !important; align-items: center !important; }
    [data-testid="stIconMaterial"] svg { font-size: 20px !important; width: 20px !important; height: 20px !important; }

    /* Force ALL text in main content area to be dark — override any inherited light colors */
    .main * { color: var(--gray-800) !important; }
    .main [data-testid="stExpander"] {
        background: white !important;
        border: 1px solid rgba(99,102,241,0.15) !important; border-radius: 20px !important;
        box-shadow: var(--shadow-lg) !important; overflow: hidden !important;
        padding: 0.25rem !important;
    }
    .main [data-testid="stExpander"] summary { background: white !important; border-radius: 16px !important; padding: 0.75rem 1rem !important; }
    .main [data-testid="stExpander"] summary,
    .main [data-testid="stExpander"] summary *,
    .main [data-testid="stExpander"] details * { color: var(--gray-800) !important; }
    .main .stMarkdown p, .main .stMarkdown li, .main .stMarkdown td,
    .main .stMarkdown span, .main .stMarkdown strong, .main .stMarkdown b,
    .main .stMarkdown em, .main .stMarkdown code { color: var(--gray-700) !important; }
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3,
    .main .stMarkdown h4, .main .stMarkdown h5 { color: var(--gray-900) !important; }
    .main .stTabs [data-baseweb="tab-panel"] * { color: var(--gray-800) !important; }
    .main .stTabs [data-baseweb="tab-panel"] h1,
    .main .stTabs [data-baseweb="tab-panel"] h2,
    .main .stTabs [data-baseweb="tab-panel"] h3,
    .main .stTabs [data-baseweb="tab-panel"] h4 { color: var(--gray-900) !important; }
    .main .stTabs [data-baseweb="tab-panel"] hr { border-color: rgba(0,0,0,0.1) !important; }
    .main label, .main .stSlider label,
    .main [data-baseweb="select"] { color: var(--gray-800) !important; }
    .main [data-testid="stWidgetLabel"] label p { color: var(--gray-800) !important; }
    /* Keep elements that SHOULD be white on dark/colored backgrounds */
    .hero-badge, .hero-badge * { color: white !important; }
    .step-number { color: white !important; }
    .score-circle, .score-circle * { color: white !important; }
    .pipe-icon, .pipe-icon * { color: white !important; }
    .onboarding-step-badge, .onboarding-step-badge * { color: white !important; }
    .stTabs [aria-selected="true"] { color: white !important; }
    .stButton button { color: white !important; }
    .stDownloadButton button { color: white !important; }

    html { scroll-behavior: smooth; }
    ::-webkit-scrollbar { width:8px; height:8px; }
    ::-webkit-scrollbar-track { background:rgba(0,0,0,0.1); border-radius:10px; }
    ::-webkit-scrollbar-thumb { background:linear-gradient(180deg, var(--primary), var(--primary-dark)); border-radius:10px; }
</style>
"""


# =============================================================================
# CORE ENGINE — Feature Extraction
# =============================================================================

class FeatureExtractor:
    """Extract per-segment features focused on endpoint connectivity."""

    def __init__(self, lines: List[LineString], precision: int = 6):
        self.lines = lines
        self.precision = precision
        self._endpoint_map: Dict[Tuple, List[int]] = defaultdict(list)
        self._build_endpoint_map()

    def _round(self, pt: Tuple) -> Tuple:
        return (round(pt[0], self.precision), round(pt[1], self.precision))

    def _build_endpoint_map(self):
        for idx, line in enumerate(self.lines):
            coords = list(line.coords)
            self._endpoint_map[self._round(coords[0])].append(idx)
            self._endpoint_map[self._round(coords[-1])].append(idx)

    def extract_all(self) -> pd.DataFrame:
        rows = []
        for idx, line in enumerate(self.lines):
            coords = list(line.coords)
            length = line.length
            n_vertices = len(coords)
            vertex_density = n_vertices / length if length > 0 else 0
            start = self._round(coords[0])
            end = self._round(coords[-1])
            start_degree = len(self._endpoint_map[start])
            end_degree = len(self._endpoint_map[end])

            # Nearest distance from each endpoint to any OTHER line
            start_pt, end_pt = Point(start), Point(end)
            min_dist_start = float('inf')
            min_dist_end = float('inf')
            nearest_seg_start = -1
            nearest_seg_end = -1
            for j, other in enumerate(self.lines):
                if j == idx:
                    continue
                d_s = start_pt.distance(other)
                d_e = end_pt.distance(other)
                if d_s < min_dist_start:
                    min_dist_start = d_s
                    nearest_seg_start = j + 1
                if d_e < min_dist_end:
                    min_dist_end = d_e
                    nearest_seg_end = j + 1
            connectivity_score = min(min_dist_start, min_dist_end)

            rows.append({
                'geometry_id': idx + 1,
                'length': round(length, 4),
                'n_vertices': n_vertices,
                'vertex_density': round(vertex_density, 6),
                'start_x': start[0], 'start_y': start[1],
                'end_x': end[0], 'end_y': end[1],
                'start_degree': start_degree,
                'end_degree': end_degree,
                'connectivity_score': round(connectivity_score, 4),
                'min_gap_start': round(min_dist_start, 4),
                'min_gap_end': round(min_dist_end, 4),
                'nearest_seg_start': nearest_seg_start,
                'nearest_seg_end': nearest_seg_end,
            })
        return pd.DataFrame(rows)


# =============================================================================
# CORE ENGINE — Gap Detector (ONE error type: endpoint gaps)
# =============================================================================

class GapDetector:
    """
    Detects the ONE error type: route continuity gaps.
    A gap exists when a dangling endpoint (degree == 1) is near but not
    touching another road segment — the coordinates don't match exactly,
    breaking route continuity.
    All thresholds derived from the dataset itself (no hardcoded values).
    """

    def detect(self, features: pd.DataFrame) -> List[Dict]:
        issues: List[Dict] = []

        # Compute adaptive gap threshold using TWO signals:
        # 1. Data-driven: 75th percentile of dangling endpoint gaps (captures most gaps)
        # 2. Scale-aware: cap at 15% of average segment length (prevents dead-end FPs)
        # The minimum of both ensures near-miss gaps are caught without
        # flagging legitimate dead-end road terminals.
        dangling_start_gaps = features.loc[features['start_degree'] == 1, 'min_gap_start']
        dangling_end_gaps = features.loc[features['end_degree'] == 1, 'min_gap_end']
        all_dangling_gaps = pd.concat([dangling_start_gaps, dangling_end_gaps])
        positive_gaps = all_dangling_gaps[(all_dangling_gaps > 0) & np.isfinite(all_dangling_gaps)]

        avg_length = float(features['length'].mean()) if len(features) > 0 else 0
        scale_threshold = avg_length * 0.15 if avg_length > 0 else 5.0

        if len(positive_gaps) >= 4:
            data_threshold = float(np.percentile(positive_gaps, 75))
            gap_threshold = min(data_threshold, scale_threshold)
        elif len(positive_gaps) >= 1:
            gap_threshold = scale_threshold
        else:
            gap_threshold = scale_threshold

        for _, row in features.iterrows():
            gid = int(row['geometry_id'])

            # Check START endpoint
            if row['start_degree'] == 1 and 0 < row['min_gap_start'] < gap_threshold:
                gap = float(row['min_gap_start'])
                issues.append({
                    'geometry_id': gid,
                    'error_type': 'ENDPOINT_GAP',
                    'endpoint': 'start',
                    'description': (
                        f"Start endpoint ({row['start_x']}, {row['start_y']}) has a "
                        f"{gap:.4f}-unit gap to nearest segment #{int(row['nearest_seg_start'])}. "
                        f"Route continuity is broken."
                    ),
                    'gap_distance': gap,
                    'gap_to_segment': int(row['nearest_seg_start']),
                    'location': (row['start_x'], row['start_y']),
                    'start': (row['start_x'], row['start_y']),
                    'end': (row['end_x'], row['end_y']),
                    'confidence': min(1.0, 1 - gap / gap_threshold),
                    'source': 'rule',
                })

            # Check END endpoint
            if row['end_degree'] == 1 and 0 < row['min_gap_end'] < gap_threshold:
                gap = float(row['min_gap_end'])
                issues.append({
                    'geometry_id': gid,
                    'error_type': 'ENDPOINT_GAP',
                    'endpoint': 'end',
                    'description': (
                        f"End endpoint ({row['end_x']}, {row['end_y']}) has a "
                        f"{gap:.4f}-unit gap to nearest segment #{int(row['nearest_seg_end'])}. "
                        f"Route continuity is broken."
                    ),
                    'gap_distance': gap,
                    'gap_to_segment': int(row['nearest_seg_end']),
                    'location': (row['end_x'], row['end_y']),
                    'start': (row['start_x'], row['start_y']),
                    'end': (row['end_x'], row['end_y']),
                    'confidence': min(1.0, 1 - gap / gap_threshold),
                    'source': 'rule',
                })

        return issues


# =============================================================================
# CORE ENGINE — ML Anomaly Detection (Isolation Forest)
# =============================================================================

class AnomalyDetector:
    """Isolation Forest to flag segments with anomalous connectivity — supports gap detection."""

    def __init__(self, contamination: float = 0.15):
        self.contamination = contamination
        self.scaler = StandardScaler()

    def detect(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        features = features.copy()

        # Guard: need enough samples for meaningful anomaly detection
        if len(features) < 5:
            features['ml_anomaly'] = 0
            features['ml_score'] = 0.0
            return features, []

        feat_cols = ['length', 'n_vertices', 'vertex_density', 'connectivity_score',
                     'start_degree', 'end_degree', 'min_gap_start', 'min_gap_end']
        X = features[feat_cols].copy()
        X = X.replace([np.inf, -np.inf], 999999)
        X_scaled = self.scaler.fit_transform(X.values)

        # Adjust contamination if dataset is small — avoid flagging too many
        effective_contamination = min(self.contamination, max(0.05, 1.0 / len(features)))
        model = IsolationForest(contamination=effective_contamination, n_estimators=100, random_state=42)
        preds = model.fit_predict(X_scaled)
        scores = model.decision_function(X_scaled)

        features['ml_anomaly'] = (preds == -1).astype(int)
        features['ml_score'] = np.round(-scores, 4)

        issues: List[Dict] = []
        for _, row in features[features['ml_anomaly'] == 1].iterrows():
            # Use the max of endpoint gaps (the more problematic endpoint)
            gap_dist = float(max(row['min_gap_start'], row['min_gap_end']))
            if np.isinf(gap_dist):
                gap_dist = float(row['connectivity_score']) if np.isfinite(row['connectivity_score']) else 0.0
            issues.append({
                'geometry_id': int(row['geometry_id']),
                'error_type': 'ENDPOINT_GAP',
                'endpoint': 'ml_flagged',
                'description': (
                    f"ML model flagged segment #{int(row['geometry_id'])} as having "
                    f"anomalous connectivity (score: {row['ml_score']:.4f}). "
                    f"Potential hidden gap or unusual endpoint pattern."
                ),
                'gap_distance': gap_dist,
                'location': (row['start_x'], row['start_y']),
                'start': (row['start_x'], row['start_y']),
                'end': (row['end_x'], row['end_y']),
                'confidence': min(1.0, float(row['ml_score']) / 0.5) if row['ml_score'] > 0 else 0.3,
                'source': 'ml',
            })
        return features, issues


# =============================================================================
# CORE ENGINE — Decision Logic
# =============================================================================

class DecisionEngine:
    @staticmethod
    def combine(rule_issues: List[Dict], ml_issues: List[Dict]) -> List[Dict]:
        """Merge, deduplicate, boost confidence for double-flagged."""
        by_id: Dict[int, List[Dict]] = defaultdict(list)
        for issue in rule_issues + ml_issues:
            by_id[issue['geometry_id']].append(issue)

        final: List[Dict] = []
        seen_keys = set()

        for gid, items in by_id.items():
            sources = set(i['source'] for i in items)
            both = 'rule' in sources and 'ml' in sources
            for item in items:
                key = (item['geometry_id'], item.get('endpoint', ''))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                entry = dict(item)
                if both:
                    entry['confidence'] = min(1.0, entry.get('confidence', 0.5) * 1.3)
                    entry['confirmed_by'] = 'rule+ml'
                else:
                    entry['confirmed_by'] = entry['source']
                c = entry['confidence']
                entry['severity'] = 'HIGH' if c >= 0.7 else ('MEDIUM' if c >= 0.4 else 'LOW')
                final.append(entry)

        final.sort(key=lambda x: -x['confidence'])
        return final


# =============================================================================
# CORE ENGINE — Auto-Fix Suggestions
# =============================================================================

class AutoFixer:
    """Suggest snapping coordinates to fix endpoint gaps."""

    def __init__(self, lines: List[LineString], precision: int = 6):
        self.lines = lines
        self.precision = precision

    def suggest_fixes(self, issues: List[Dict]) -> List[Dict]:
        suggestions = []
        for issue in issues:
            if issue.get('endpoint') == 'ml_flagged':
                continue
            gid = issue['geometry_id'] - 1
            if gid >= len(self.lines):
                continue
            line = self.lines[gid]
            coords = list(line.coords)
            is_start = issue.get('endpoint') == 'start'
            coord_idx = 0 if is_start else -1
            pt = Point(coords[coord_idx])

            best_dist = float('inf')
            best_snap = None
            best_target = None
            for j, other in enumerate(self.lines):
                if j == gid:
                    continue
                nearest = other.interpolate(other.project(pt))
                d = pt.distance(nearest)
                if 0 < d < best_dist:
                    best_dist = d
                    best_snap = (round(nearest.x, self.precision), round(nearest.y, self.precision))
                    best_target = j + 1

            if best_snap:
                original = (round(coords[coord_idx][0], self.precision),
                            round(coords[coord_idx][1], self.precision))
                suggestions.append({
                    'geometry_id': issue['geometry_id'],
                    'fix_type': 'SNAP_ENDPOINT',
                    'endpoint': issue.get('endpoint', 'unknown'),
                    'original_coord': original,
                    'suggested_coord': best_snap,
                    'snap_to_segment': best_target,
                    'distance': round(best_dist, 4),
                    'description': (
                        f"Snap {issue.get('endpoint','')} endpoint from "
                        f"({original[0]}, {original[1]}) → ({best_snap[0]}, {best_snap[1]}) "
                        f"to close {best_dist:.4f}-unit gap"
                    ),
                })
        return suggestions


# =============================================================================
# CORE ENGINE — Report Builder
# =============================================================================

def build_error_report(issues: List[Dict], fixes: Optional[List[Dict]] = None) -> Dict:
    report_items = []
    for issue in issues:
        report_items.append({
            'geometry_id': issue['geometry_id'],
            'error_type': issue['error_type'],
            'endpoint': issue.get('endpoint', ''),
            'severity': issue.get('severity', 'MEDIUM'),
            'confidence': round(issue.get('confidence', 0.5), 4),
            'description': issue.get('description', ''),
            'gap_distance': issue.get('gap_distance', 0),
            'coordinates': {
                'start': list(issue.get('start', (0, 0))),
                'end': list(issue.get('end', (0, 0))),
            },
            'confirmed_by': issue.get('confirmed_by', issue.get('source', 'rule')),
        })
    return {
        'report_version': APP_CONFIG['version'],
        'team': APP_CONFIG['team'],
        'error_type_focus': APP_CONFIG['error_type'],
        'total_gaps_found': len(report_items),
        'issues': report_items,
        'auto_fix_suggestions': fixes or [],
    }


def generate_text_report(stats: Dict, issues: List[Dict], fixes: List[Dict]) -> bytes:
    out = []
    out.append("=" * 72)
    out.append("ROUTE CONTINUITY GAP DETECTION REPORT")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"Team: {APP_CONFIG['team']} | Version: {APP_CONFIG['version']}")
    out.append(f"Error Type: {APP_CONFIG['error_label']}")
    out.append("=" * 72)
    out.append("")
    out.append("NETWORK SUMMARY")
    out.append("-" * 40)
    out.append(f"  Total Segments:    {stats['total_segments']}")
    out.append(f"  Total Length:      {stats['total_length']:,.2f}")
    out.append(f"  Connected Nodes:   {stats['connected_nodes']}")
    out.append(f"  Dangling Nodes:    {stats['dangling_nodes']}")
    out.append("")
    out.append(f"GAPS DETECTED: {len(issues)}")
    out.append("-" * 40)
    if issues:
        for idx, issue in enumerate(issues, 1):
            out.append(f"  [{idx}] Segment #{issue['geometry_id']} ({issue.get('endpoint','')}) — {issue.get('severity','')}")
            out.append(f"      Gap: {issue.get('gap_distance',0):.4f} units | Confidence: {issue.get('confidence',0):.0%}")
            out.append(f"      {issue.get('description','')}")
            out.append("")
    else:
        out.append("  No gaps detected. Route continuity is intact.")
        out.append("")
    if fixes:
        out.append(f"FIX SUGGESTIONS: {len(fixes)}")
        out.append("-" * 40)
        for idx, fix in enumerate(fixes, 1):
            out.append(f"  [{idx}] {fix['description']}")
            out.append("")
    out.append("=" * 72)
    return "\n".join(out).encode('utf-8')


# =============================================================================
# WKT PARSER
# =============================================================================

def parse_wkt(wkt_text: str) -> List[LineString]:
    cleaned = re.sub(r'\s+', ' ', wkt_text)
    matches = re.findall(r'LINESTRING\s*\([^)]+\)', cleaned, re.IGNORECASE)
    lines = []
    for m in matches:
        try:
            geom = wkt.loads(m)
            if isinstance(geom, LineString) and geom.is_valid and not geom.is_empty:
                lines.append(geom)
        except Exception:
            continue
    return lines


# =============================================================================
# MAP VISUALIZATION
# =============================================================================

def create_map(lines: List[LineString], issues: List[Dict]) -> folium.Map:
    if not lines:
        return folium.Map(location=[0, 0], zoom_start=2, tiles='cartodbpositron')

    all_x, all_y = [], []
    for line in lines:
        for c in line.coords:
            all_x.append(c[0]); all_y.append(c[1])
    cx, cy = np.mean(all_x), np.mean(all_y)
    scale = 0.0001

    def norm(x, y):
        return [(y - cy) * scale, (x - cx) * scale]

    m = folium.Map(location=[0, 0], zoom_start=15, tiles=None)
    folium.TileLayer('cartodbdark_matter', name='🌙 Dark Mode').add_to(m)
    folium.TileLayer('cartodbpositron', name='☀️ Light Mode').add_to(m)

    flagged_ids = set(i['geometry_id'] for i in issues)
    severity_map = {}
    for i in issues:
        gid = i['geometry_id']
        if gid not in severity_map or i.get('severity') == 'HIGH':
            severity_map[gid] = i.get('severity', 'MEDIUM')

    road_group = folium.FeatureGroup(name='🛣️ Road Network')
    gap_group = folium.FeatureGroup(name='⚠️ Gap Segments')
    for idx, line in enumerate(lines):
        gid = idx + 1
        coords = [norm(c[0], c[1]) for c in line.coords]
        if gid in flagged_ids:
            sev = severity_map.get(gid, 'MEDIUM')
            color = '#ef4444' if sev == 'HIGH' else '#f59e0b' if sev == 'MEDIUM' else '#60a5fa'
            folium.PolyLine(coords, color=color, weight=4, opacity=0.95,
                tooltip=f"Seg #{gid} — GAP ({sev})").add_to(gap_group)
        else:
            folium.PolyLine(coords, color='#60a5fa', weight=2.5, opacity=0.7,
                tooltip=f"Seg #{gid}").add_to(road_group)
    road_group.add_to(m)
    gap_group.add_to(m)

    if issues:
        marker_group = folium.FeatureGroup(name='📍 Gap Locations')
        for issue in issues:
            loc_point = issue.get('location', issue.get('start', (cx, cy)))
            loc = norm(loc_point[0], loc_point[1])
            sev = issue.get('severity', 'MEDIUM')
            color = '#ef4444' if sev == 'HIGH' else '#f59e0b' if sev == 'MEDIUM' else '#94a3b8'

            # Build human-readable "why flagged" explanation
            gap_dist = issue.get('gap_distance', 0)
            endpoint_label = issue.get('endpoint', '')
            gap_to = issue.get('gap_to_segment', '?')
            conf = issue.get('confidence', 0)
            src = issue.get('confirmed_by', issue.get('source', 'rule'))

            if endpoint_label == 'ml_flagged':
                why_text = (
                    f"The ML model detected that this segment has unusual "
                    f"connectivity compared to the rest of the network. "
                    f"Its endpoint distances and topology deviate from the norm."
                )
            else:
                why_text = (
                    f"This segment's {endpoint_label} endpoint is only {gap_dist:.4f} units "
                    f"away from segment #{gap_to}, but they don't share an exact coordinate. "
                    f"In a valid road network, connecting segments must share the exact same "
                    f"endpoint coordinates — otherwise routing algorithms can't traverse the junction."
                )

            src_text = {
                'rule': 'Rule-based gap detection',
                'ml': 'Isolation Forest ML model',
                'rule+ml': 'Both rule engine AND ML model agree',
            }.get(src, src)

            folium.CircleMarker(location=loc, radius=16, color=color, fill=True,
                fillColor=color, fillOpacity=0.2, weight=0).add_to(marker_group)
            folium.CircleMarker(location=loc, radius=8, color='white', weight=2,
                fill=True, fillColor=color, fillOpacity=0.95,
                popup=folium.Popup(
                    f"""<div style="font-family:'DM Sans',sans-serif;min-width:260px;padding:4px;">
                    <h4 style="color:{color};margin:0 0 8px;font-weight:700;">🔗 Route Gap Detected</h4>
                    <div style="background:#f8fafc;border-radius:8px;padding:8px;">
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Segment:</b> #{issue['geometry_id']}</p>
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Endpoint:</b> {endpoint_label}</p>
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Gap Size:</b> {gap_dist:.4f} units</p>
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Confidence:</b> {conf:.0%}</p>
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Severity:</b>
                            <span style="background:{color};color:white;padding:1px 8px;border-radius:12px;font-size:0.75rem;">{sev}</span></p>
                        <p style="margin:4px 0;color:#334155;font-size:0.85rem;"><b>Source:</b> {src_text}</p>
                    </div>
                    <div style="background:#fef3c7;border-radius:8px;padding:8px;margin-top:8px;border-left:3px solid #f59e0b;">
                        <p style="margin:0;color:#92400e;font-size:0.8rem;font-weight:600;">⚠️ Why is this an error?</p>
                        <p style="margin:4px 0 0;color:#78350f;font-size:0.78rem;line-height:1.4;">{why_text}</p>
                    </div></div>""", max_width=320),
                tooltip=f"#{issue['geometry_id']} Gap ({sev}) — click for details"
            ).add_to(marker_group)
        marker_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    all_coords = [norm(c[0], c[1]) for line in lines for c in line.coords]
    m.fit_bounds(all_coords, padding=[30, 30])

    # Inject DM Sans font into folium map so popups and controls use it
    font_link = '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">'
    layer_css = """<style>
    .leaflet-control-layers,
    .leaflet-popup-content,
    .leaflet-tooltip {
        font-family: 'DM Sans', sans-serif !important;
    }
    .leaflet-control-layers {
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
        padding: 12px 16px !important;
        background: rgba(255,255,255,0.97) !important;
        backdrop-filter: blur(8px) !important;
    }
    .leaflet-control-layers-separator {
        border-top: 1px solid #e2e8f0 !important;
        margin: 8px 0 !important;
    }
    .leaflet-control-layers label {
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: #334155 !important;
        padding: 3px 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 4px !important;
        cursor: pointer !important;
        transition: color 0.15s !important;
    }
    .leaflet-control-layers label:hover {
        color: #6366f1 !important;
    }
    .leaflet-control-layers-toggle {
        width: 40px !important;
        height: 40px !important;
        border-radius: 12px !important;
        background-color: rgba(255,255,255,0.95) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    .leaflet-tooltip {
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1) !important;
        padding: 6px 12px !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
    }
    </style>"""
    m.get_root().header.add_child(folium.Element(font_link))
    m.get_root().header.add_child(folium.Element(layer_css))

    # Add hover highlight/thicken effect for all polylines via JavaScript
    hover_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            var map = Object.values(window).find(v => v instanceof L.Map);
            if (!map) return;
            map.eachLayer(function(layer) {
                if (layer instanceof L.Polyline && !(layer instanceof L.Polygon)) {
                    var origWeight = layer.options.weight || 3;
                    var origOpacity = layer.options.opacity || 0.7;
                    layer.on('mouseover', function() {
                        this.setStyle({weight: origWeight + 5, opacity: 1.0});
                        this.bringToFront();
                    });
                    layer.on('mouseout', function() {
                        this.setStyle({weight: origWeight, opacity: origOpacity});
                    });
                }
            });
        }, 500);
    });
    </script>
    """
    m.get_root().html.add_child(folium.Element(hover_js))

    return m


# =============================================================================
# COMPUTE STATS
# =============================================================================

def compute_stats(lines: List[LineString]) -> Dict:
    lengths = [l.length for l in lines]
    ep_count: Dict[Tuple, int] = defaultdict(int)
    prec = APP_CONFIG['precision']
    for line in lines:
        c = list(line.coords)
        s = (round(c[0][0], prec), round(c[0][1], prec))
        e = (round(c[-1][0], prec), round(c[-1][1], prec))
        ep_count[s] += 1
        ep_count[e] += 1
    return {
        'total_segments': len(lines),
        'total_length': round(sum(lengths), 2),
        'avg_length': round(np.mean(lengths), 2),
        'min_length': round(min(lengths), 4),
        'max_length': round(max(lengths), 2),
        'total_endpoints': len(ep_count),
        'connected_nodes': sum(1 for c in ep_count.values() if c > 1),
        'dangling_nodes': sum(1 for c in ep_count.values() if c == 1),
    }


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_hero():
    st.markdown("""
        <div class="hero-container">
            <div class="hero-icon">🔍</div>
            <h1 class="hero-title">Route Continuity Gap Detector</h1>
            <p class="hero-subtitle">Detects endpoint gaps where road segments should connect but don't — breaking route continuity</p>
            <div class="hero-badge"><span>🔗</span><span>ONE Error Type • Rule-Based + ML • Axes Systems GmbH</span></div>
            <p style="font-size:0.85rem;color:var(--gray-500);margin:0.6rem 0 0;">Built by <b>Puranjay Gambhir</b>, <b>Akshobhya Rao</b> & <b>Rohan</b> — IIT Mandi Hackathon 3.0, 2026</p>
        </div>
    """, unsafe_allow_html=True)


def render_metrics(stats: Dict, issues: List[Dict]):
    n_gaps = len(issues)
    high = sum(1 for i in issues if i.get('severity') == 'HIGH')
    dangling = stats['dangling_nodes']
    st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card purple"><div class="metric-icon">🛣️</div><div class="metric-value">{stats['total_segments']}</div><div class="metric-label">Segments</div></div>
            <div class="metric-card red"><div class="metric-icon">🔗</div><div class="metric-value">{n_gaps}</div><div class="metric-label">Gaps Found</div></div>
            <div class="metric-card blue"><div class="metric-icon">🚨</div><div class="metric-value">{high}</div><div class="metric-label">High Severity</div></div>
            <div class="metric-card green"><div class="metric-icon">📍</div><div class="metric-value">{dangling}</div><div class="metric-label">Dangling Nodes</div></div>
        </div>
    """, unsafe_allow_html=True)


def render_issue_table(issues: List[Dict]):
    if not issues:
        st.markdown("""
            <div style="text-align:center;padding:2.5rem;background:linear-gradient(135deg,rgba(16,185,129,0.1),rgba(5,150,105,0.1));border-radius:16px;">
                <div style="font-size:3.5rem;margin-bottom:0.75rem;">✅</div>
                <h3 style="color:#059669;margin:0 0 0.4rem;font-weight:700;">No Gaps Detected!</h3>
                <p style="color:#64748b;margin:0;">Route continuity is intact — all endpoints connect properly.</p>
            </div>
        """, unsafe_allow_html=True)
        return
    rows = []
    for i in issues:
        rows.append({
            'Seg #': i['geometry_id'],
            'Endpoint': i.get('endpoint', ''),
            'Gap (units)': f"{i.get('gap_distance', 0):.4f}",
            'Severity': i.get('severity', 'MEDIUM'),
            'Confidence': f"{i.get('confidence', 0):.0%}",
            'Source': i.get('confirmed_by', i.get('source', '')),
            'Description': i.get('description', ''),
        })
    df = pd.DataFrame(rows)
    df.index = df.index + 1
    df.index.name = '#'

    def style_sev(val):
        if val == 'HIGH': return 'background:#fef2f2;color:#991b1b;font-weight:600;'
        elif val == 'MEDIUM': return 'background:#fffbeb;color:#92400e;font-weight:600;'
        return 'background:#f0f9ff;color:#1e40af;font-weight:600;'

    col_config = {
        'Description': st.column_config.TextColumn('Description', width='large'),
    }
    st.markdown("""<div style="overflow-x:auto;max-width:100%;">""", unsafe_allow_html=True)
    st.dataframe(df.style.map(style_sev, subset=['Severity']),
                 use_container_width=True, height=420, column_config=col_config)
    st.markdown("""</div>""", unsafe_allow_html=True)


def render_stats(stats: Dict, issues: List[Dict]):
    st.markdown("""
        <div class="stats-grid">
            <div class="stat-card">
                <h4>📏 Network Metrics</h4>
                <div class="stat-item"><span class="stat-label">Total Length</span><span class="stat-value">{:,.2f}</span></div>
                <div class="stat-item"><span class="stat-label">Avg Segment</span><span class="stat-value">{:.2f}</span></div>
                <div class="stat-item"><span class="stat-label">Shortest</span><span class="stat-value">{:.4f}</span></div>
                <div class="stat-item"><span class="stat-label">Longest</span><span class="stat-value">{:.2f}</span></div>
            </div>
            <div class="stat-card">
                <h4>🔗 Topology</h4>
                <div class="stat-item"><span class="stat-label">Total Nodes</span><span class="stat-value">{}</span></div>
                <div class="stat-item"><span class="stat-label">Connected</span><span class="stat-value">{}</span></div>
                <div class="stat-item"><span class="stat-label">Dangling</span><span class="stat-value">{}</span></div>
                <div class="stat-item"><span class="stat-label">Segments</span><span class="stat-value">{}</span></div>
            </div>
        </div>
    """.format(
        stats['total_length'], stats['avg_length'], stats['min_length'], stats['max_length'],
        stats['total_endpoints'], stats['connected_nodes'], stats['dangling_nodes'], stats['total_segments']
    ), unsafe_allow_html=True)

    if stats['total_segments'] > 0:
        affected = len(set(i['geometry_id'] for i in issues))
        quality = max(0, (1 - affected / stats['total_segments']) * 100)
        sc = "excellent" if quality >= 85 else "good" if quality >= 60 else "poor"
        sl = "Excellent" if quality >= 85 else "Good" if quality >= 60 else "Needs Attention"
        se = "🌟" if quality >= 85 else "👍" if quality >= 60 else "🔧"
        st.markdown(f"""
            <div class="quality-score">
                <h3 style="margin:0 0 0.75rem;color:#1e293b;font-weight:700;">Route Continuity Score</h3>
                <div class="score-circle {sc}"><span>{quality:.0f}%</span></div>
                <div class="score-label">{se} {sl}</div>
            </div>
        """, unsafe_allow_html=True)

        # "How is this score calculated?" toggle
        with st.container():
            if st.button("❓ How is this score calculated?", key="toggle_score_explain", use_container_width=True):
                st.session_state['show_score_explain'] = not st.session_state.get('show_score_explain', False)
            if st.session_state.get('show_score_explain', False):
                st.markdown(f"""
                <div style="background:white;border-radius:16px;padding:1.25rem 1.5rem;margin-top:0.5rem;
                            border:1px solid rgba(99,102,241,0.12);box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                    <h4 style="color:#1e293b;margin:0 0 0.75rem;font-weight:700;">📐 Score Calculation</h4>
                    <p style="color:#475569;font-size:0.88rem;line-height:1.7;margin:0 0 0.75rem;">
                        The <b>Route Continuity Score</b> measures what percentage of your road
                        segments have clean, unbroken endpoint connections.
                    </p>
                    <div style="background:#f1f5f9;border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.75rem;">
                        <p style="color:#334155;font-size:0.9rem;font-weight:600;margin:0 0 0.5rem;">Formula:</p>
                        <code style="font-size:0.88rem;color:#4f46e5;background:none;">
                            Score = (1 − affected_segments / total_segments) × 100%
                        </code>
                    </div>
                    <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
                        <tr style="border-bottom:1px solid #e2e8f0;">
                            <td style="padding:0.4rem 0;color:#475569;"><b>Total Segments</b></td>
                            <td style="padding:0.4rem 0;color:#1e293b;text-align:right;font-weight:600;">{stats['total_segments']}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #e2e8f0;">
                            <td style="padding:0.4rem 0;color:#475569;"><b>Segments with Gaps</b></td>
                            <td style="padding:0.4rem 0;color:#ef4444;text-align:right;font-weight:600;">{affected}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #e2e8f0;">
                            <td style="padding:0.4rem 0;color:#475569;"><b>Clean Segments</b></td>
                            <td style="padding:0.4rem 0;color:#10b981;text-align:right;font-weight:600;">{stats['total_segments'] - affected}</td>
                        </tr>
                        <tr>
                            <td style="padding:0.4rem 0;color:#475569;"><b>Continuity Score</b></td>
                            <td style="padding:0.4rem 0;color:#4f46e5;text-align:right;font-weight:700;font-size:1rem;">{quality:.0f}%</td>
                        </tr>
                    </table>
                    <p style="color:#64748b;font-size:0.8rem;margin:0.75rem 0 0;line-height:1.5;">
                        <b>🌟 Excellent (≥85%)</b> — minimal gaps, network is well-connected<br>
                        <b>👍 Good (60–84%)</b> — some gaps exist but network is mostly functional<br>
                        <b>🔧 Needs Attention (&lt;60%)</b> — significant connectivity issues detected
                    </p>
                </div>
                """, unsafe_allow_html=True)


def render_examples():
    """Demonstrate the error type on 2-3 built-in examples."""
    st.markdown("""
        <div class="section-header"><span class="icon">🧪</span><h3>Error Examples — Demonstration</h3></div>
        <p style="color:#64748b;margin-bottom:1rem;font-size:0.9rem;">
            Below are built-in examples showing <b>correct</b> road connections vs. <b>incorrect</b> ones with endpoint gaps.
            Click "Run Analysis" on any example to see the detector in action.
        </p>
    """, unsafe_allow_html=True)

    # Define examples list
    examples = [
        {
            'key': 'correct',
            'wkt': EXAMPLE_CORRECT,
            'card_class': 'correct',
            'icon': '✅',
            'title': 'Example: Correct — Fully Connected Network',
            'title_color': '#059669',
            'desc': 'All 5 segments share exact endpoints. No gaps detected.',
        },
        {
            'key': 'error1',
            'wkt': EXAMPLE_ERROR_1,
            'card_class': 'error',
            'icon': '❌',
            'title': 'Example 1: Two Endpoint Gaps (~0.5 units)',
            'title_color': '#ef4444',
            'desc': 'Segments 1→2 and 3→5 have small coordinate mismatches at their junction points.',
        },
        {
            'key': 'error2',
            'wkt': EXAMPLE_ERROR_2,
            'card_class': 'error',
            'icon': '❌',
            'title': 'Example 2: Multiple Gaps in a Chain',
            'title_color': '#ef4444',
            'desc': 'A 6-segment chain with gaps at segments 3, 4, and 6 — breaking 3 continuity points.',
        },
        {
            'key': 'error3',
            'wkt': EXAMPLE_ERROR_3,
            'card_class': 'error',
            'icon': '❌',
            'title': 'Example 3: Near-Miss Gaps (0.05–0.6 units)',
            'title_color': '#ef4444',
            'desc': 'Endpoints are very close but not touching — typical of coordinate rounding errors during data export.',
        },
    ]

    for ex in examples:
        st.markdown(f"""
            <div class="example-card {ex['card_class']}">
                <h4 style="color:{ex['title_color']};margin:0 0 0.5rem;">{ex['icon']} {ex['title']}</h4>
                <p style="color:#475569;font-size:0.85rem;margin:0;">{ex['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Show WKT in a styled container instead of st.expander
        with st.container():
            show_key = f"show_wkt_{ex['key']}"
            if st.button(f"📄 View WKT & Run Analysis — {ex['title'].split(':')[0] if ':' in ex['title'] else 'Example'}", key=f"toggle_{ex['key']}", use_container_width=True):
                st.session_state[show_key] = not st.session_state.get(show_key, False)

            if st.session_state.get(show_key, False):
                st.code(ex['wkt'], language="text")
                ex_key = ex['key']
                if st.button(f"▶ Run on {ex['title'].split(':')[0] if ':' in ex['title'] else ex['title']}", key=f"run_{ex_key}"):
                    st.session_state[f'example_result_{ex_key}'] = ex_key

            # Show persisted results from session state
            if st.session_state.get(f'example_result_{ex["key"]}') == ex['key']:
                _run_example(ex['wkt'], ex['key'])


def _run_example(wkt_data: str, label: str):
    """Analyse an example dataset inline — results persist via session state."""
    lines = parse_wkt(wkt_data)
    if not lines:
        st.error("Could not parse example.")
        return
    extractor = FeatureExtractor(lines, APP_CONFIG['precision'])
    features = extractor.extract_all()
    detector = GapDetector()
    issues = detector.detect(features)

    if issues:
        st.error(f"🔗 **{len(issues)} gap(s) detected!**")
        for i, issue in enumerate(issues, 1):
            st.markdown(f"**Gap {i}:** Segment #{issue['geometry_id']} ({issue.get('endpoint','')}) — "
                        f"gap of **{issue.get('gap_distance',0):.4f}** units — "
                        f"confidence {issue.get('confidence',0):.0%}")
    else:
        st.success("✅ **No gaps detected.** All endpoints connect properly.")

    map_obj = create_map(lines, issues)
    st_folium(map_obj, height=350, use_container_width=True, returned_objects=[])
    if st.button("✕ Clear Result", key=f"clear_{label}"):
        del st.session_state[f'example_result_{label}']
        st.rerun()


def render_training():
    """Training instructions — styled cards matching the app aesthetic."""
    st.markdown("""
        <div class="section-header"><span class="icon">📚</span><h3>Training Instructions</h3></div>
    """, unsafe_allow_html=True)

    # Intro card
    st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(139,92,246,0.08));
                    border-radius:20px;padding:1.5rem 1.75rem;margin-bottom:1.5rem;
                    border:1px solid rgba(99,102,241,0.12);">
            <h4 style="color:#4f46e5;margin:0 0 0.5rem;font-weight:700;">🤖 No Manual Labeling Needed</h4>
            <p style="color:#475569;font-size:0.9rem;line-height:1.7;margin:0;">
                The system uses <b>unsupervised learning</b> (Isolation Forest) — it learns what "normal"
                looks like from whatever dataset you upload. However, you can improve detection by
                adding more example data. Follow the steps below.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Training steps as styled cards
    training_steps = [
        {
            'num': '1', 'color': '#6366f1', 'icon': '📄',
            'title': 'Prepare Your WKT File',
            'body': (
                'Create a <code>.wkt</code> text file with LINESTRING geometries — one per line:'
                '<div style="background:#f1f5f9;border-radius:10px;padding:0.75rem 1rem;margin:0.6rem 0;'
                'font-family:monospace;font-size:0.85rem;color:#334155;line-height:1.8;">'
                'LINESTRING(x1 y1, x2 y2, x3 y3, ...)<br>'
                'LINESTRING(x4 y4, x5 y5, x6 y6, ...)'
                '</div>'
                'Each LINESTRING represents a road segment. Coordinates can be in any projected CRS.'
            ),
        },
        {
            'num': '2', 'color': '#f59e0b', 'icon': '🔗',
            'title': 'Include Known Errors',
            'body': (
                'To test the system, deliberately introduce endpoint gaps:'
                '<ul style="margin:0.5rem 0;padding-left:1.5rem;line-height:1.8;">'
                '<li>Take two segments that should connect: <code>...140 220)</code> and <code>LINESTRING(140 220, ...)</code></li>'
                '<li>Shift one endpoint slightly: <code>LINESTRING(140.5 220.3, ...)</code></li>'
                '<li>This creates a gap the detector will flag</li>'
                '</ul>'
            ),
        },
        {
            'num': '3', 'color': '#10b981', 'icon': '📤',
            'title': 'Upload and Analyze',
            'body': (
                '<ol style="margin:0.5rem 0;padding-left:1.5rem;line-height:1.8;">'
                '<li>Use the <b>sidebar file uploader</b> to load your <code>.wkt</code> file</li>'
                '<li>The system automatically extracts features, runs rules + ML, and reports gaps</li>'
                '<li>Adjust the <b>Anomaly Sensitivity</b> slider to control ML aggressiveness</li>'
                '</ol>'
            ),
        },
        {
            'num': '4', 'color': '#ef4444', 'icon': '📋',
            'title': 'Interpret Results',
            'body': (
                '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin:0.5rem 0;">'
                '<tr style="border-bottom:2px solid #e2e8f0;"><th style="text-align:left;padding:0.4rem 0.5rem;color:#1e293b;">Severity</th>'
                '<th style="text-align:left;padding:0.4rem 0.5rem;color:#1e293b;">Meaning</th></tr>'
                '<tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:0.4rem 0.5rem;"><span style="background:#fef2f2;color:#991b1b;padding:0.15rem 0.5rem;border-radius:6px;font-weight:600;font-size:0.8rem;">HIGH</span></td>'
                '<td style="padding:0.4rem 0.5rem;color:#475569;">Clearly a broken connection (confidence ≥ 70%)</td></tr>'
                '<tr style="border-bottom:1px solid #f1f5f9;"><td style="padding:0.4rem 0.5rem;"><span style="background:#fffbeb;color:#92400e;padding:0.15rem 0.5rem;border-radius:6px;font-weight:600;font-size:0.8rem;">MEDIUM</span></td>'
                '<td style="padding:0.4rem 0.5rem;color:#475569;">Likely a gap, could be intentional (40–70%)</td></tr>'
                '<tr><td style="padding:0.4rem 0.5rem;"><span style="background:#f0f9ff;color:#1e40af;padding:0.15rem 0.5rem;border-radius:6px;font-weight:600;font-size:0.8rem;">LOW</span></td>'
                '<td style="padding:0.4rem 0.5rem;color:#475569;">Possible gap but less certain (&lt;40%)</td></tr>'
                '</table>'
                '<div style="margin-top:0.6rem;font-size:0.85rem;color:#475569;line-height:1.7;">'
                '<b>Source: rule</b> — detected by the data-driven gap rule<br>'
                '<b>Source: ml</b> — detected by Isolation Forest anomaly model<br>'
                '<b>Source: rule+ml</b> — flagged by both (highest reliability)'
                '</div>'
            ),
        },
        {
            'num': '5', 'color': '#8b5cf6', 'icon': '🔧',
            'title': 'Download the Auto-Fix',
            'body': (
                'Go to the <b>🔧 Auto-Fix</b> tab to download a corrected <code>.wkt</code> file '
                'with all detected gaps snapped closed. Compare the original and corrected files '
                'to verify the changes.'
            ),
        },
    ]

    for s in training_steps:
        st.markdown(f"""
            <div style="background:white;border-radius:18px;padding:1.25rem 1.5rem;margin-bottom:1rem;
                        border:1px solid rgba(99,102,241,0.1);border-left:4px solid {s['color']};
                        box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.6rem;">
                    <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,{s['color']},{s['color']}cc);
                                display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;color:white !important;">
                        {s['icon']}
                    </div>
                    <div>
                        <span style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{s['color']};">Step {s['num']}</span>
                        <h4 style="margin:0;font-size:1rem;font-weight:700;color:#1e293b;">{s['title']}</h4>
                    </div>
                </div>
                <div style="color:#475569;font-size:0.88rem;line-height:1.7;">
                    {s['body']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # How the ML learns — highlight card
    st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(5,150,105,0.08));
                    border-radius:18px;padding:1.25rem 1.5rem;margin-top:0.5rem;
                    border:1px solid rgba(16,185,129,0.15);">
            <h4 style="color:#059669;margin:0 0 0.5rem;font-weight:700;">🧠 How the ML Learns</h4>
            <p style="color:#475569;font-size:0.88rem;line-height:1.7;margin:0 0 0.5rem;">
                The Isolation Forest is <b>retrained on every upload</b> — it learns the statistical
                distribution of your specific dataset's features (length, vertex density, connectivity).
                Larger and more varied datasets produce more reliable anomaly detection.
            </p>
            <p style="color:#166534;font-size:0.85rem;font-weight:600;margin:0;">
                ✅ No saved model or manual labels needed — the system adapts automatically.
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_welcome():
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🔍</div>
            <h2 class="welcome-title">Route Continuity Gap Detector</h2>
            <p class="welcome-text">
                Detects <b>one specific error</b>: endpoint gaps where road segments
                should connect but have small coordinate mismatches, breaking route
                continuity. Uses rule-based validation + Isolation Forest ML.
            </p>
            <div class="step-list">
                <div class="step-item"><div class="step-number">1</div><span class="step-text">Upload a .wkt file or load the demo dataset</span></div>
                <div class="step-item"><div class="step-number">2</div><span class="step-text">One-click analysis finds all endpoint gaps</span></div>
                <div class="step-item"><div class="step-number">3</div><span class="step-text">View gaps on the interactive map</span></div>
                <div class="step-item"><div class="step-number">4</div><span class="step-text">Download auto-fixed .wkt or error report</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    render_info_sections()
    render_onboarding()


def render_info_sections():
    """Supported format card + How It Works pipeline — shown on welcome page and below analysis."""
    # Supported format
    st.markdown("""
        <div style="background:white;border-radius:20px;padding:1.5rem 2rem;margin-top:1.25rem;
                    border:1px solid rgba(99,102,241,0.12);box-shadow:0 4px 12px rgba(0,0,0,0.06);">
            <h4 style="color:#1e293b;margin:0 0 0.75rem;font-weight:700;">📖 Supported Format</h4>
            <div style="background:#f1f5f9;border-radius:12px;padding:1rem 1.25rem;font-family:monospace;font-size:0.88rem;color:#334155;line-height:1.7;">
                LINESTRING(x1 y1, x2 y2, x3 y3, ...)<br>
                LINESTRING(x1 y1, x2 y2, ...)
            </div>
            <p style="color:#475569;font-size:0.9rem;margin:0.75rem 0 0;">Each line is a valid WKT LINESTRING. Coordinates in any projected CRS.</p>
        </div>
    """, unsafe_allow_html=True)

    # How It Works pipeline
    st.markdown("""
        <div class="how-it-works">
            <h2 class="hiw-title">⚙️ How It Works</h2>
            <p class="hiw-subtitle">Our 5-stage AI pipeline detects broken road connections automatically</p>
            <div class="pipeline-steps">
                <div class="pipe-step">
                    <div class="pipe-icon a">📄</div>
                    <div class="pipe-content">
                        <p class="pipe-label">Stage A</p>
                        <p class="pipe-title">Parse WKT Input</p>
                        <p class="pipe-desc">Your <code>.wkt</code> file is parsed into individual LINESTRING road segments. Each segment's start/end coordinates are extracted for analysis.</p>
                    </div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-icon b">📊</div>
                    <div class="pipe-content">
                        <p class="pipe-label">Stage B</p>
                        <p class="pipe-title">Extract Connectivity Features</p>
                        <p class="pipe-desc">For each segment, we compute: endpoint degree (how many roads share that point), distance to nearest neighbor, segment length, and vertex density. These features reveal connectivity patterns.</p>
                    </div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-icon c">🔍</div>
                    <div class="pipe-content">
                        <p class="pipe-label">Stage C</p>
                        <p class="pipe-title">Rule-Based Gap Detection</p>
                        <p class="pipe-desc">Dangling endpoints (connected to only 1 segment) are checked: if they're <i>near</i> another road but don't share an exact coordinate, that's a gap. The threshold adapts to each dataset — no hardcoded values.</p>
                    </div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-icon d">🤖</div>
                    <div class="pipe-content">
                        <p class="pipe-label">Stage D</p>
                        <p class="pipe-title">ML Anomaly Detection</p>
                        <p class="pipe-desc">An Isolation Forest model learns what "normal" connectivity looks like from your data. Segments with unusual endpoint patterns are flagged as potential hidden gaps. No training labels needed.</p>
                    </div>
                </div>
                <div class="pipe-step">
                    <div class="pipe-icon e">✅</div>
                    <div class="pipe-content">
                        <p class="pipe-label">Stage E</p>
                        <p class="pipe-title">Decision & Auto-Fix</p>
                        <p class="pipe-desc">Rule-based and ML results are merged. Segments flagged by <i>both</i> get a 30% confidence boost. Each gap gets a severity rating (HIGH/MEDIUM/LOW) and an auto-fix suggestion with exact snap coordinates.</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_onboarding():
    """Interactive step-by-step tutorial with progress bar and visual icons."""
    st.markdown("<br>", unsafe_allow_html=True)

    if 'onboarding_step' not in st.session_state:
        st.session_state['onboarding_step'] = 0

    step = st.session_state['onboarding_step']

    tutorial_steps = [
        {
            'icon': '📁', 'color': '#6366f1',
            'title': 'Upload Your Data',
            'desc': (
                'Look at the <b>sidebar on the left</b>. You\'ll see a file uploader — '
                'drag and drop your <code>.wkt</code> file there. Or click the '
                '<b>"🎯 Load Demo Data"</b> button to try with our built-in 56-segment street network. '
                'This is where everything starts!'
            ),
        },
        {
            'icon': '📊', 'color': '#f59e0b',
            'title': 'Check the Dashboard Metrics',
            'desc': (
                'Once data is loaded, four metric cards appear at the top: <b>Segments</b> (total road count), '
                '<b>Gaps Found</b> (detected errors), <b>High Severity</b> (critical gaps), and '
                '<b>Dangling Nodes</b> (endpoints connected to only one road). '
                'A quick glance tells you the overall health of your network.'
            ),
            'tip': '💡 <b>Pro tip:</b> Dangling nodes aren\'t always errors — some are legitimate dead-end roads.',
        },
        {
            'icon': '🗺️', 'color': '#10b981',
            'title': 'Explore the Interactive Map',
            'desc': (
                'The <b>Map tab</b> shows your road network interactively. '
                '<b>Blue lines</b> = healthy roads. <b>Red/orange lines</b> = segments with gaps. '
                '<b>Colored circle markers</b> = exact gap locations — click them for a detailed popup '
                'explaining <i>what</i> the gap is, <i>how big</i> it is, and <i>why</i> it was flagged. '
                '<b>Hover over any segment</b> to highlight it. '
                'Use the <b>layer panel</b> (top-right) to toggle layers on/off.'
            ),
            'tip': '💡 <b>Pro tip:</b> Click any red circle marker for the full error explanation and fix suggestion.',
        },
        {
            'icon': '📋', 'color': '#ef4444',
            'title': 'Review the Gap Report',
            'desc': (
                'Switch to the <b>Gap Report tab</b> to see a sortable table of every detected gap: '
                'which segment, which endpoint, the gap distance, severity (HIGH/MEDIUM/LOW), '
                'and confidence percentage. You can <b>download</b> the report as CSV, JSON, or a text file.'
            ),
            'tip': '💡 <b>Pro tip:</b> Click column headers to sort — try sorting by Confidence to see the most certain gaps first.',
        },
        {
            'icon': '🔧', 'color': '#8b5cf6',
            'title': 'Download the Auto-Fix',
            'desc': (
                'Go to the <b>Auto-Fix tab</b> to see exactly how each gap should be corrected. '
                'The system calculates the nearest snap point on the neighboring segment and shows '
                'the original vs. corrected coordinates. Click <b>"Download Corrected .wkt"</b> '
                'to get a fixed version of your road network with all gaps snapped closed!'
            ),
            'tip': '💡 <b>Pro tip:</b> Compare the original and corrected files in a diff tool to verify changes.',
        },
    ]

    total = len(tutorial_steps)

    if step < total:
        s = tutorial_steps[step]
        pct = ((step + 1) / total) * 100

        # Progress bar + step dots
        dots_html = ""
        for i in range(total):
            if i < step:
                dots_html += f'<div style="width:12px;height:12px;border-radius:50%;background:#10b981;"></div>'
            elif i == step:
                dots_html += f'<div style="width:12px;height:12px;border-radius:50%;background:{s["color"]};box-shadow:0 0 0 3px {s["color"]}44;"></div>'
            else:
                dots_html += '<div style="width:12px;height:12px;border-radius:50%;background:#e2e8f0;"></div>'

        tip_html = ""
        if s.get('tip'):
            tip_html = (
                '<div style="background:#f0fdf4;border-radius:10px;padding:0.6rem 0.9rem;border-left:3px solid #10b981;">'
                f'<p style="color:#166534;font-size:0.82rem;margin:0;line-height:1.5;">{s["tip"]}</p>'
                '</div>'
            )

        # Build the full HTML as a regular string to avoid f-string/markdown truncation
        card_html = (
            '<div class="onboarding-overlay">'
            f'<div class="onboarding-step-badge">Step {step + 1} of {total}</div>'
            f'<div style="display:flex;align-items:center;gap:0.75rem;margin:0.75rem 0 0.5rem;">'
            f'<div style="width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,{s["color"]},{s["color"]}cc);display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">{s["icon"]}</div>'
            f'<h3 style="margin:0;font-size:1.15rem;font-weight:700;color:#1e293b;">{s["title"]}</h3>'
            '</div>'
            f'<p style="color:#475569;font-size:0.9rem;line-height:1.6;margin:0 0 0.75rem;">{s["desc"]}</p>'
            + tip_html +
            f'<div style="margin-top:1rem;height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;">'
            f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{s["color"]},{s["color"]}cc);border-radius:3px;transition:width 0.3s;"></div>'
            '</div>'
            f'<div style="display:flex;justify-content:center;gap:8px;margin-top:0.6rem;">{dots_html}</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if step > 0:
                if st.button("← Previous", key="onboard_prev"):
                    st.session_state['onboarding_step'] = step - 1
                    st.rerun()
        with col2:
            if st.button("Skip Tutorial", key="onboard_skip"):
                st.session_state['onboarding_step'] = total
                st.rerun()
        with col3:
            if step < total - 1:
                if st.button("Next →", key="onboard_next"):
                    st.session_state['onboarding_step'] = step + 1
                    st.rerun()
            else:
                if st.button("✅ Finish", key="onboard_finish"):
                    st.session_state['onboarding_step'] = total
                    st.rerun()
    else:
        # Tutorial completed — show a small restart link
        if st.button("🔄 Restart Tutorial", key="onboard_restart"):
            st.session_state['onboarding_step'] = 0
            st.rerun()


def render_sidebar() -> Tuple[Optional[str], float]:
    with st.sidebar:
        st.markdown(f"""
            <div style="text-align:center;padding:1rem 0;">
                <div style="font-size:2.5rem;margin-bottom:0.4rem;">🔍</div>
                <h2 style="margin:0;font-weight:800;font-size:1.4rem;">Gap Detector</h2>
                <p style="margin:0.2rem 0 0;font-size:0.8rem;opacity:0.7;">v{APP_CONFIG['version']} • {APP_CONFIG['team']}</p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📁 Data Input")
        uploader_key = st.session_state.get('uploader_key', 0)
        uploaded = st.file_uploader("Upload .wkt / .txt", type=['wkt', 'txt'],
            help="LINESTRING geometries", key=f"file_uploader_{uploader_key}")
        use_demo = st.button("🎯 Load Demo Data (56 segs)", type="primary", use_container_width=True)
        if uploaded is not None or st.session_state.get('data_source'):
            if st.button("🔄 Clear Loaded Data", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key.startswith('example_result_'):
                        del st.session_state[key]
                st.session_state['data_source'] = None
                st.session_state['uploaded_wkt'] = None
                st.session_state['uploader_key'] = uploader_key + 1
                st.rerun()

        st.divider()
        st.markdown("### ⚙️ ML Sensitivity")
        contamination = st.slider("Anomaly Sensitivity", 0.05, 0.30, 0.15, 0.01,
            help="Higher = flag more segments as anomalous")

        st.markdown("""
        <div style="font-size:0.82rem;line-height:1.55;background:rgba(99,102,241,0.08);border-radius:10px;padding:0.7rem 0.85rem;margin-top:0.4rem;border-left:3px solid #6366f1;">
            <b style="color:#a5b4fc;">What this means</b><br>
            <span style="opacity:0.85;">Controls how aggressively the Isolation Forest ML model flags segments as anomalous.</span>
            <ul style="margin:0.4rem 0 0;padding-left:1.1rem;opacity:0.85;">
                <li><b>Low (0.05)</b> — only the most obvious outliers are flagged</li>
                <li><b>Default (0.15)</b> — balanced precision &amp; recall</li>
                <li><b>High (0.30)</b> — catches more gaps but may include false positives</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 🔗 Error Type Focus")
        st.markdown("""
        <div style="font-size:0.85rem;line-height:1.6;background:rgba(239,68,68,0.1);border-radius:12px;padding:0.75rem;">
            <b style="color:#ef4444;">ENDPOINT GAP</b><br>
            <span style="opacity:0.8;">Detects where road endpoints should connect but have coordinate gaps, breaking route continuity.</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
            <div style="text-align:center;font-size:0.72rem;opacity:0.6;">
                FutureFormers • IIT Mandi Hackathon 3.0<br>
                Puranjay · Akshobhya · Rohan<br>
                Axes Systems GmbH • Option B<br>© 2026
            </div>
        """, unsafe_allow_html=True)

    wkt_data = None
    if use_demo:
        wkt_data = DEMO_WKT_DATA
        st.session_state['data_source'] = 'demo'
        st.session_state['uploaded_wkt'] = None
    elif uploaded is not None:
        wkt_data = uploaded.read().decode('utf-8')
        st.session_state['data_source'] = 'upload'
        st.session_state['uploaded_wkt'] = wkt_data
    elif st.session_state.get('data_source') == 'demo':
        wkt_data = DEMO_WKT_DATA
    elif st.session_state.get('data_source') == 'upload' and st.session_state.get('uploaded_wkt'):
        wkt_data = st.session_state['uploaded_wkt']

    return wkt_data, contamination


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon=APP_CONFIG['icon'],
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = None

    wkt_data, contamination = render_sidebar()
    render_hero()

    if wkt_data:
        # 1. Parse
        lines = parse_wkt(wkt_data)
        if not lines:
            st.error("No valid LINESTRING geometries found in the uploaded file.")
            return

        # Data source badge
        source = st.session_state.get('data_source', '')
        if source == 'demo':
            st.markdown(f"""<div style="text-align:center;margin-bottom:1rem;">
                <span style="background:linear-gradient(135deg,#6366f1,#4f46e5);color:white;padding:0.35rem 1.2rem;border-radius:100px;font-size:0.82rem;font-weight:600;letter-spacing:0.02em;">
                📁 Demo Dataset — {len(lines)} Street Segments</span></div>""", unsafe_allow_html=True)
        elif source == 'upload':
            st.markdown(f"""<div style="text-align:center;margin-bottom:1rem;">
                <span style="background:linear-gradient(135deg,#10b981,#059669);color:white;padding:0.35rem 1.2rem;border-radius:100px;font-size:0.82rem;font-weight:600;letter-spacing:0.02em;">
                📄 Uploaded File — {len(lines)} Segments Parsed</span></div>""", unsafe_allow_html=True)

        with st.spinner(f"🔍 Analyzing {len(lines)} segments for endpoint gaps..."):
            # 2. Feature Extraction
            extractor = FeatureExtractor(lines, APP_CONFIG['precision'])
            features = extractor.extract_all()

            # 3. Gap Detection (rule-based)
            gap_detector = GapDetector()
            rule_issues = gap_detector.detect(features)

            # 4. ML Anomaly Detection
            ml_detector = AnomalyDetector(contamination=contamination)
            features, ml_issues = ml_detector.detect(features)

            # 5. Decision Logic
            all_issues = DecisionEngine.combine(rule_issues, ml_issues)

            # 6. Auto-fix suggestions
            fixer = AutoFixer(lines, APP_CONFIG['precision'])
            fixes = fixer.suggest_fixes(all_issues)

            # 7. Stats & Report
            stats = compute_stats(lines)
            report = build_error_report(all_issues, fixes)

        render_metrics(stats, all_issues)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🗺️ Map",
            "📋 Gap Report",
            "🔧 Auto-Fix",
            "🧪 Examples",
            "📚 Training",
            "📊 Statistics",
            "⚙️ How It Works",
        ])

        with tab1:
            st.markdown("""
                <div class="section-header"><span class="icon">🗺️</span><h3>Network Map — Gap Visualization</h3></div>
                <p style="color:#64748b;margin-bottom:0.75rem;font-size:0.9rem;">
                    Red/Orange = gap segments • Click markers for gap details • Toggle layers top-right
                </p>
            """, unsafe_allow_html=True)
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            st_folium(create_map(lines, all_issues), height=550, use_container_width=True, returned_objects=[])
            st.markdown('</div>', unsafe_allow_html=True)

            # Map layer legend / explanation
            st.markdown("""
                <div class="legend-box">
                    <p class="legend-title">🗂️ Map Layer Guide — What Each Control Means</p>
                    <div class="legend-grid">
                        <div class="legend-item">
                            <div class="legend-label"><b>🌙 Dark Mode</b> — Dark base map theme for better contrast with colored road segments</div>
                        </div>
                        <div class="legend-item">
                            <div class="legend-label"><b>☀️ Light Mode</b> — Light/minimal base map theme, easier on the eyes for detailed inspection</div>
                        </div>
                        <div class="legend-item">
                            <div class="legend-label"><b>🛣️ Road Network</b> — All healthy road segments with no detected gaps (shown in blue)</div>
                        </div>
                        <div class="legend-item">
                            <div class="legend-label"><b>⚠️ Gap Segments</b> — Road segments that have a broken endpoint connection (red = high, orange = medium severity)</div>
                        </div>
                        <div class="legend-item">
                            <div class="legend-label"><b>📍 Gap Locations</b> — Circle markers at exact gap positions. Click for details on why it's an error and how to fix it</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("""<div class="section-header"><span class="icon">🔗</span><h3>Detected Endpoint Gaps</h3></div>""", unsafe_allow_html=True)
            render_issue_table(all_issues)
            if all_issues:
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = pd.DataFrame([{
                        'Segment': i['geometry_id'], 'Endpoint': i.get('endpoint',''),
                        'Gap': i.get('gap_distance',0), 'Severity': i.get('severity',''),
                        'Confidence': i.get('confidence',0), 'Source': i.get('confirmed_by',''),
                        'Description': i.get('description','')
                    } for i in all_issues]).to_csv(index=False)
                    st.download_button("📥 CSV", csv, "gap_report.csv", "text/csv", use_container_width=True)
                with col2:
                    st.download_button("📥 JSON", json.dumps(report, indent=2),
                        "error_report.json", "application/json", use_container_width=True)
                with col3:
                    st.download_button("📥 Report (.txt)",
                        generate_text_report(stats, all_issues, fixes),
                        "gap_report.txt", "text/plain", use_container_width=True)

        with tab3:
            st.markdown("""<div class="section-header"><span class="icon">🔧</span><h3>Auto-Fix — Snap Endpoints</h3></div>""", unsafe_allow_html=True)
            if fixes:
                st.markdown("""<p style="color:#64748b;margin-bottom:0.75rem;font-size:0.9rem;">
                    Suggested coordinate corrections to close detected gaps.
                </p>""", unsafe_allow_html=True)
                fix_df = pd.DataFrame([{
                    'Seg #': f['geometry_id'],
                    'Endpoint': f['endpoint'],
                    'Original': f"({f['original_coord'][0]}, {f['original_coord'][1]})",
                    'Fix To': f"({f['suggested_coord'][0]}, {f['suggested_coord'][1]})",
                    'Snap To Seg': f"#{f['snap_to_segment']}",
                    'Gap': f"{f['distance']:.4f}",
                } for f in fixes])
                fix_df.index = fix_df.index + 1
                st.dataframe(fix_df, use_container_width=True, height=300)

                st.markdown("---")
                corrected_lines = list(lines)
                for fix in fixes:
                    gid = fix['geometry_id'] - 1
                    if gid >= len(corrected_lines):
                        continue
                    coords = list(corrected_lines[gid].coords)
                    if fix['endpoint'] == 'start':
                        coords[0] = fix['suggested_coord']
                    else:
                        coords[-1] = fix['suggested_coord']
                    corrected_lines[gid] = LineString(coords)
                corrected_wkt = "\n".join(l.wkt for l in corrected_lines)
                st.download_button("📥 Download Corrected .wkt", corrected_wkt,
                    "corrected_network.wkt", "text/plain", use_container_width=True)
            else:
                st.markdown("""
                    <div style="text-align:center;padding:2.5rem;background:linear-gradient(135deg,rgba(16,185,129,0.1),rgba(5,150,105,0.1));border-radius:16px;">
                        <div style="font-size:3.5rem;margin-bottom:0.75rem;">✅</div>
                        <h3 style="color:#059669;margin:0 0 0.4rem;font-weight:700;">No Fixes Needed</h3>
                        <p style="color:#64748b;margin:0;">No endpoint gaps requiring correction.</p>
                    </div>
                """, unsafe_allow_html=True)

        with tab4:
            render_examples()

        with tab5:
            render_training()

        with tab6:
            st.markdown("""<div class="section-header"><span class="icon">📊</span><h3>Network Statistics</h3></div>""", unsafe_allow_html=True)
            render_stats(stats, all_issues)

            with st.container():
                if st.button("📊 View Extracted Feature Data", key="toggle_feat_data", use_container_width=True):
                    st.session_state['show_feat_data'] = not st.session_state.get('show_feat_data', False)

                if st.session_state.get('show_feat_data', False):
                    st.markdown("""<p style="color:#64748b;font-size:0.85rem;margin-bottom:0.75rem;">
                        Per-segment features used by the rule engine and ML model. Scroll right to see all columns.
                    </p>""", unsafe_allow_html=True)
                    display_cols = ['geometry_id', 'length', 'n_vertices', 'vertex_density',
                                    'start_degree', 'end_degree', 'connectivity_score',
                                    'min_gap_start', 'min_gap_end']
                    if 'ml_anomaly' in features.columns:
                        display_cols += ['ml_anomaly', 'ml_score']
                    feat_display = features[display_cols].copy()
                    feat_display.index = feat_display.index + 1

                    def highlight_anomaly(val):
                        if val == 1:
                            return 'background:#fef2f2;color:#991b1b;font-weight:600;'
                        return ''

                    if 'ml_anomaly' in feat_display.columns:
                        st.dataframe(feat_display.style.map(highlight_anomaly, subset=['ml_anomaly']),
                                     use_container_width=True, height=400)
                    else:
                        st.dataframe(feat_display, use_container_width=True, height=400)

        with tab7:
            st.markdown("""<div class="section-header"><span class="icon">⚙️</span><h3>How It Works</h3></div>""", unsafe_allow_html=True)
            render_info_sections()

        # Tutorial stays visible below analysis tabs
        render_onboarding()

    else:
        render_welcome()


if __name__ == "__main__":
    main()
