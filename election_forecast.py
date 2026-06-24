import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as pe
import json, sys
from pathlib import Path

st.set_page_config(page_title="Election Forecast — SD4", page_icon="🗳️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; }
  .section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase; color: #888;
    border-bottom: 1px solid #e0e0e0; padding-bottom: 0.4rem; margin-bottom: 1rem;
  }
  .stat-card { background: #f7f7f5; border-left: 3px solid #1a6b3c; padding: 0.85rem 1rem; border-radius: 2px; margin-bottom: 0.5rem; }
  .stat-card .label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #666; margin-bottom: 0.2rem; }
  .stat-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #111; line-height: 1; }
  .stat-card .sub { font-size: 0.72rem; color: #888; margin-top: 0.2rem; }
  .win-hero { background: linear-gradient(135deg, #0f3d24 0%, #1a6b3c 100%); color: white; padding: 1.5rem 1.75rem; border-radius: 4px; text-align: center; }
  .win-hero .wlabel { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.7; margin-bottom: 0.5rem; }
  .win-hero .wvalue { font-family: 'IBM Plex Mono', monospace; font-size: 3.2rem; font-weight: 600; line-height: 1; }
  .win-hero .wsub { font-size: 0.75rem; opacity: 0.6; margin-top: 0.5rem; }
  .styled-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .styled-table th { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.06em; text-transform: uppercase; color: #666; padding: 0.4rem 0.6rem; border-bottom: 2px solid #ddd; text-align: right; }
  .styled-table th:first-child { text-align: left; }
  .styled-table td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #f0f0f0; text-align: right; font-variant-numeric: tabular-nums; }
  .styled-table td:first-child { text-align: left; font-weight: 500; }
  .styled-table tr:hover td { background: #fafaf8; }
  .prob-high { color: #1a6b3c; font-weight: 600; }
  .prob-mid  { color: #d97706; font-weight: 600; }
  .prob-low  { color: #b91c1c; font-weight: 600; }
  .model-box { background: #f7f7f5; border: 1px solid #e8e8e4; border-radius: 4px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
  .model-box h4 { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #1a6b3c; margin: 0 0 0.6rem 0; }
  .model-box p { font-size: 0.85rem; color: #444; line-height: 1.6; margin: 0 0 0.5rem 0; }
  .model-box code { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; background: #eee; padding: 0.1rem 0.3rem; border-radius: 2px; }
  .poll-blend { background: #f0f7f3; border: 1px solid #c3dece; border-radius: 4px; padding: 1rem 1.25rem; margin-bottom: 1rem; }
  .context-badge { display: inline-block; background: #1a6b3c; color: white; font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; padding: 0.2rem 0.5rem; border-radius: 2px; letter-spacing: 0.06em; margin-right: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── Geographic data (SD4 boundary + helpers) ──────────────────────────────────
COUNTY_CENTROIDS = {
    "Alpine":(-119.73,38.60),"Amador":(-120.65,38.43),"Calaveras":(-120.56,38.14),
    "El Dorado":(-120.52,38.78),"Inyo":(-117.69,36.66),"Madera":(-119.71,37.22),
    "Mariposa":(-119.91,37.57),"Merced":(-120.65,37.23),"Mono":(-118.96,37.94),
    "Nevada":(-120.83,39.27),"Placer":(-120.74,38.90),"Stanislaus":(-120.70,37.47),
    "Tuolumne":(-119.94,37.86),
}
SD4_COUNTIES = set(COUNTY_CENTROIDS.keys())

DATA_DIR = Path(__file__).parent

# ── SD4 boundary (from uploaded GeoJSON, simplified with RDP epsilon=0.005) ──
SD4_BOUNDARY = [(-116.47215, 36.44656), (-115.64837, 35.80922), (-115.73576, 35.8091), (-115.7359, 35.79363), (-117.91927, 35.7984), (-117.92302, 35.78669), (-118.00811, 35.78894), (-118.00774, 35.81709), (-117.99862, 35.82303), (-118.00621, 35.82904), (-118.00743, 35.85819), (-117.99671, 35.8695), (-117.98077, 35.86752), (-117.98908, 35.88131), (-117.98206, 35.8947), (-117.99033, 35.91405), (-117.9833, 35.92657), (-117.99196, 35.94378), (-118.0168, 35.95456), (-118.01458, 35.9729), (-118.00359, 35.98372), (-118.01209, 35.99831), (-118.03362, 36.00895), (-118.05173, 36.05955), (-118.05163, 36.08355), (-118.06733, 36.09345), (-118.07353, 36.14035), (-118.05973, 36.15085), (-118.05933, 36.17015), (-118.10563, 36.21344), (-118.10523, 36.23364), (-118.11938, 36.25557), (-118.1174, 36.27117), (-118.12761, 36.28035), (-118.12702, 36.30023), (-118.11183, 36.30834), (-118.11253, 36.32204), (-118.09763, 36.33114), (-118.10033, 36.34614), (-118.12423, 36.35194), (-118.13053, 36.37034), (-118.16293, 36.38944), (-118.14033, 36.40354), (-118.14123, 36.42094), (-118.15683, 36.43264), (-118.19363, 36.42654), (-118.21443, 36.43434), (-118.20963, 36.44214), (-118.21563, 36.45634), (-118.24993, 36.48244), (-118.23503, 36.49374), (-118.24143, 36.50004), (-118.23883, 36.52364), (-118.24943, 36.52404), (-118.25203, 36.54214), (-118.26534, 36.55154), (-118.29124, 36.55934), (-118.28904, 36.59074), (-118.27464, 36.59734), (-118.32134, 36.62744), (-118.31884, 36.63894), (-118.33824, 36.65544), (-118.33124, 36.66944), (-118.34754, 36.67234), (-118.36614, 36.69044), (-118.33554, 36.70444), (-118.33604, 36.71634), (-118.35094, 36.74074), (-118.36924, 36.75004), (-118.38024, 36.78224), (-118.37404, 36.80018), (-118.39374, 36.82967), (-118.36218, 36.84401), (-118.37019, 36.87169), (-118.36084, 36.88774), (-118.38844, 36.94554), (-118.40475, 36.95754), (-118.40445, 36.97204), (-118.41955, 36.98744), (-118.41225, 36.99834), (-118.42799, 37.01124), (-118.42277, 37.02581), (-118.44174, 37.04502), (-118.43498, 37.04974), (-118.43715, 37.05982), (-118.44897, 37.06914), (-118.46734, 37.06675), (-118.50312, 37.09523), (-118.52214, 37.09836), (-118.53094, 37.11119), (-118.5831, 37.1224), (-118.59267, 37.13815), (-118.65461, 37.14183), (-118.67266, 37.16735), (-118.66425, 37.17815), (-118.66647, 37.1895), (-118.68136, 37.20444), (-118.67563, 37.21415), (-118.68644, 37.22758), (-118.68376, 37.24437), (-118.66534, 37.26192), (-118.71603, 37.32821), (-118.74113, 37.31534), (-118.78675, 37.34339), (-118.78153, 37.35642), (-118.76816, 37.36118), (-118.79004, 37.39404), (-118.77977, 37.42168), (-118.76928, 37.42218), (-118.76019, 37.43364), (-118.7633, 37.45654), (-118.79711, 37.48872), (-118.85606, 37.4784), (-118.86077, 37.50154), (-118.90187, 37.52604), (-118.91707, 37.55034), (-118.92967, 37.54894), (-118.95267, 37.56584), (-118.97697, 37.55684), (-119.02208, 37.58584), (-119.3122, 37.35273), (-119.31224, 37.33971), (-119.32579, 37.33542), (-119.31576, 37.32478), (-119.33544, 37.31122), (-119.32695, 37.29191), (-119.33232, 37.27444), (-119.32219, 37.25342), (-119.3378, 37.21917), (-119.33009, 37.20743), (-119.34312, 37.18908), (-119.36058, 37.18054), (-119.36193, 37.16785), (-119.38886, 37.14922), (-119.41581, 37.16345), (-119.43214, 37.16258), (-119.43489, 37.14696), (-119.46265, 37.14422), (-119.47482, 37.10997), (-119.49133, 37.11972), (-119.48919, 37.13727), (-119.50606, 37.15035), (-119.51748, 37.14678), (-119.52499, 37.12824), (-119.55001, 37.14459), (-119.56085, 37.14291), (-119.56858, 37.11669), (-119.54877, 37.11676), (-119.5376, 37.10507), (-119.55901, 37.08806), (-119.56144, 37.06549), (-119.60492, 37.07102), (-119.62113, 37.02661), (-119.63538, 37.02155), (-119.62905, 37.03462), (-119.64947, 37.0435), (-119.65939, 37.03894), (-119.65862, 37.01334), (-119.69809, 37.00875), (-119.74051, 36.97022), (-119.74327, 36.95464), (-119.73378, 36.94645), (-119.75207, 36.93591), (-119.75496, 36.92246), (-119.77275, 36.9186), (-119.78932, 36.89671), (-119.78613, 36.8789), (-119.81876, 36.84807), (-119.8409, 36.8613), (-119.855, 36.85118), (-119.8849, 36.85855), (-119.91193, 36.84532), (-119.92911, 36.84791), (-119.94352, 36.83404), (-119.97055, 36.83287), (-119.98466, 36.84085), (-119.9924, 36.82895), (-120.01369, 36.82814), (-120.02779, 36.81451), (-120.07947, 36.82535), (-120.11491, 36.81434), (-120.11034, 36.87301), (-120.05597, 36.87312), (-120.05602, 36.91662), (-120.03758, 36.91665), (-120.03243, 36.93542), (-119.98754, 36.93786), (-120.01304, 36.96707), (-120.01056, 36.97432), (-119.99821, 36.97419), (-120.00092, 36.99451), (-120.02972, 36.98465), (-120.02415, 36.99163), (-120.02904, 37.0326), (-120.0017, 37.0328), (-120.00207, 37.07045), (-120.02934, 37.06675), (-120.02935, 37.0836), (-120.02207, 37.08358), (-120.03918, 37.10147), (-120.0347, 37.1116), (-120.05552, 37.12479), (-120.09456, 37.12859), (-120.12033, 37.1475), (-120.12154, 37.15911), (-120.10917, 37.1659), (-120.05207, 37.18311), (-120.09009, 37.22148), (-120.14384, 37.2392), (-120.17851, 37.26243), (-120.18772, 37.30108), (-120.28341, 37.42437), (-120.27498, 37.44364), (-120.28401, 37.4629), (-120.3116, 37.45561), (-120.37022, 37.42229), (-120.44951, 37.4011), (-120.45527, 37.41498), (-120.46889, 37.40599), (-120.4698, 37.41356), (-120.48257, 37.41839), (-120.48718, 37.40403), (-120.52135, 37.40392), (-120.52135, 37.41843), (-120.57598, 37.4184), (-120.57616, 37.4039), (-120.66786, 37.40391), (-120.66842, 37.36717), (-120.70969, 37.38231), (-120.70048, 37.38405), (-120.70479, 37.40044), (-120.72849, 37.40816), (-120.74263, 37.39934), (-120.74134, 37.39262), (-120.74607, 37.39925), (-120.76728, 37.38821), (-120.79175, 37.39229), (-120.82816, 37.38154), (-120.83575, 37.36921), (-120.86218, 37.35763), (-120.88109, 37.36275), (-120.89696, 37.35558), (-120.90773, 37.36364), (-120.9221, 37.35557), (-120.92409, 37.36969), (-120.94015, 37.3691), (-120.94686, 37.35897), (-120.95896, 37.35793), (-120.95651, 37.34917), (-121.22682, 37.13478), (-121.23712, 37.15721), (-121.26211, 37.15933), (-121.28112, 37.18361), (-121.29856, 37.16597), (-121.32842, 37.16595), (-121.36095, 37.18433), (-121.38428, 37.16621), (-121.38356, 37.15149), (-121.39903, 37.15014), (-121.41341, 37.17233), (-121.40881, 37.18251), (-121.42184, 37.22131), (-121.44176, 37.23113), (-121.45576, 37.24944), (-121.45908, 37.28232), (-121.44966, 37.29394), (-121.42347, 37.29529), (-121.40577, 37.31099), (-121.42366, 37.35884), (-121.40909, 37.38068), (-121.42406, 37.39364), (-121.45666, 37.39554), (-121.45636, 37.40674), (-121.47262, 37.42335), (-121.46187, 37.4388), (-121.46293, 37.45149), (-121.48679, 37.47566), (-121.2409, 37.66478), (-121.2302, 37.66298), (-121.21853, 37.67358), (-121.22348, 37.68361), (-121.20871, 37.68631), (-121.20246, 37.69599), (-121.18119, 37.6879), (-121.17987, 37.70415), (-121.16369, 37.70038), (-121.15511, 37.72033), (-121.12137, 37.72192), (-121.10764, 37.73229), (-121.11006, 37.74213), (-121.09538, 37.73323), (-121.0877, 37.74141), (-121.067, 37.73957), (-121.05675, 37.75052), (-121.044, 37.73871), (-121.02883, 37.74071), (-121.01773, 37.75528), (-121.00801, 37.7491), (-120.99345, 37.76095), (-120.95404, 37.73836), (-120.9216, 37.73765), (-120.91722, 37.75221), (-120.92341, 37.75803), (-120.92646, 38.07743), (-120.93886, 38.08833), (-121.0271, 38.30026), (-121.02744, 38.50815), (-121.11906, 38.71787), (-121.13317, 38.70538), (-121.14161, 38.71194), (-121.13452, 38.71204), (-121.11859, 38.76894), (-121.10176, 38.78798), (-121.10177, 38.81523), (-121.08497, 38.81602), (-121.08735, 38.83328), (-121.05842, 38.84713), (-121.06182, 38.85992), (-121.05328, 38.86835), (-121.06197, 38.88214), (-121.0443, 38.89034), (-121.05355, 38.89881), (-121.04042, 38.91566), (-121.00026, 38.91795), (-120.95778, 38.93911), (-120.93858, 38.93557), (-120.93622, 38.96393), (-120.90197, 38.95312), (-120.88794, 38.95974), (-120.85988, 38.95166), (-120.84991, 38.97655), (-120.8347, 38.9716), (-120.82474, 38.99275), (-120.8086, 39.001), (-120.78853, 38.99953), (-120.78599, 39.02772), (-120.80207, 39.02657), (-120.76942, 39.05876), (-120.79026, 39.07152), (-120.78273, 39.08071), (-120.7629, 39.07549), (-120.77072, 39.09034), (-120.76876, 39.10744), (-120.7351, 39.11911), (-120.71903, 39.11444), (-120.70166, 39.13593), (-120.68359, 39.13775), (-120.6924, 39.12193), (-120.68952, 39.11335), (-120.6703, 39.1116), (-120.67044, 39.09505), (-120.66519, 39.10168), (-120.63864, 39.10101), (-120.6171, 39.1162), (-120.63456, 39.11633), (-120.62161, 39.13458), (-120.61477, 39.18185), (-120.59859, 39.18724), (-120.56745, 39.19286), (-120.56882, 39.18328), (-120.5362, 39.18375), (-120.50563, 39.1735), (-120.50283, 39.15642), (-120.471, 39.17426), (-120.46049, 39.16582), (-120.43963, 39.17716), (-120.42561, 39.1999), (-120.43258, 39.21672), (-120.4439, 39.22268), (-120.43407, 39.24172), (-120.42064, 39.24348), (-120.41658, 39.25271), (-120.45391, 39.25865), (-120.44506, 39.29231), (-120.45733, 39.30554), (-120.44524, 39.30755), (-120.46445, 39.31614), (-120.46124, 39.32099), (-120.40793, 39.32905), (-120.40014, 39.33671), (-120.37389, 39.32702), (-120.33479, 39.34344), (-120.3059, 39.33587), (-120.27335, 39.35817), (-120.25472, 39.35868), (-120.25485, 39.37397), (-120.23654, 39.37415), (-120.22042, 39.36313), (-120.18169, 39.37259), (-120.16094, 39.36729), (-120.14936, 39.37473), (-120.11624, 39.36799), (-120.08587, 39.38733), (-120.0643, 39.37376), (-120.04098, 39.37043), (-120.02305, 39.37984), (-120.02116, 39.40014), (-120.00528, 39.40008), (-120.00103, 38.99958), (-118.92253, 38.24992), (-118.0522, 37.62494), (-117.24493, 37.03025), (-116.47215, 36.44656)]

def _rdp(points, epsilon):
    import sys as _sys; _sys.setrecursionlimit(20000)
    def inner(pts, eps):
        if len(pts)<3: return list(pts)
        x1,y1=float(pts[0][0]),float(pts[0][1]); x2,y2=float(pts[-1][0]),float(pts[-1][1])
        dx,dy=x2-x1,y2-y1; md,mi=0,0
        for i in range(1,len(pts)-1):
            x0,y0=float(pts[i][0]),float(pts[i][1])
            if dx==0 and dy==0: d=((x0-x1)**2+(y0-y1)**2)**0.5
            else:
                t=max(0,min(1,((x0-x1)*dx+(y0-y1)*dy)/(dx*dx+dy*dy)))
                d=((x0-(x1+t*dx))**2+(y0-(y1+t*dy))**2)**0.5
            if d>md: md,mi=d,i
        if md>eps: return inner(pts[:mi+1],eps)[:-1]+inner(pts[mi:],eps)
        return [pts[0],pts[-1]]
    return inner(points, epsilon)

@st.cache_data(show_spinner="Loading county boundaries…", ttl=86400)
def load_county_geojson():
    import urllib.request, json as _json
    urls = [
        "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson",
        "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/california-counties.geojson",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                data = _json.loads(r.read())
            polys = {}
            for feat in data.get("features", []):
                name = (feat.get("properties",{}).get("name") or
                        feat.get("properties",{}).get("NAME") or "").replace(" County","").strip()
                if name in SD4_COUNTIES:
                    geom = feat["geometry"]
                    rings = [geom["coordinates"][0]] if geom["type"]=="Polygon" else [p[0] for p in geom["coordinates"]]
                    ring  = max(rings, key=len)
                    polys[name] = _rdp(ring, 0.002)
            if len(polys) >= 10:
                return polys
        except Exception:
            continue
    return {}

# ── Forecast model integration ────────────────────────────────────────────────
def load_forecast_params():
    """Load the most recent forecast_params.json if it exists."""
    p = DATA_DIR / "forecast_params.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def params_to_county_dict(params: dict) -> dict:
    """Convert forecast_params county entries to the COUNTIES dict format."""
    result = {}
    for cn, d in params["counties"].items():
        result[cn] = {
            "registration": int(d["registration"]),
            "turnout":      float(d["turnout"]),
            "lean_avg":     float(d["lean_avg"]),
            "lean_lin":     float(d["lean_lin"]),
            "lean_sd":      float(d["lean_sd"]),
            "hist_avg":     float(d["hist_avg"]),
        }
    return result

def params_to_county_df(counties: dict) -> pd.DataFrame:
    rows = []
    for cn, d in counties.items():
        rows.append({
            "County":       cn,
            "Registration": int(d["registration"]),
            "Turnout (%)":  round(d["turnout"]  * 100, 2),
            "Lean Avg (%)": round(d["lean_avg"] * 100, 2),
            "Lean Lin (%)": round(d["lean_lin"] * 100, 2),
            "Lean SD (%)":  round(d["lean_sd"]  * 100, 2),
            "Hist Avg (%)": round(d["hist_avg"] * 100, 2),
        })
    return pd.DataFrame(rows)

def run_forecast_model(ctx: dict) -> dict:
    """Run forecast_model.py programmatically and return params."""
    try:
        if str(DATA_DIR) not in sys.path:
            sys.path.insert(0, str(DATA_DIR))
        import importlib
        import forecast_model as fm
        importlib.reload(fm)
        return fm.run_forecast(ctx, verbose=False)
    except Exception as e:
        st.error(f"Forecast model error: {e}")
        return None

# ── Hardcoded fallback defaults (used only if forecast_params.json absent) ───
FALLBACK_COUNTIES = {
    "Alpine":     {"registration":944,   "turnout":0.7074,"lean_avg":0.0388,"lean_lin":0.0995,"lean_sd":0.0293,"hist_avg":0.6221},
    "Amador":     {"registration":27416, "turnout":0.7110,"lean_avg":-0.2465,"lean_lin":-0.2444,"lean_sd":0.0143,"hist_avg":0.3481},
    "Calaveras":  {"registration":33312, "turnout":0.6702,"lean_avg":-0.2404,"lean_lin":-0.2538,"lean_sd":0.0162,"hist_avg":0.3581},
    "El Dorado":  {"registration":142947,"turnout":0.6630,"lean_avg":-0.1784,"lean_lin":-0.1531,"lean_sd":0.0140,"hist_avg":0.4079},
    "Inyo":       {"registration":11037, "turnout":0.6890,"lean_avg":-0.1280,"lean_lin":-0.0973,"lean_sd":0.0122,"hist_avg":0.4570},
    "Madera":     {"registration":34903, "turnout":0.5690,"lean_avg":-0.2700,"lean_lin":-0.2325,"lean_sd":0.0182,"hist_avg":0.3155},
    "Mariposa":   {"registration":11862, "turnout":0.6941,"lean_avg":-0.2083,"lean_lin":-0.2038,"lean_sd":0.0065,"hist_avg":0.3772},
    "Merced":     {"registration":12842, "turnout":0.4989,"lean_avg":-0.2673,"lean_lin":-0.2239,"lean_sd":0.0231,"hist_avg":0.3182},
    "Mono":       {"registration":8286,  "turnout":0.6654,"lean_avg":-0.0251,"lean_lin":-0.0033,"lean_sd":0.0173,"hist_avg":0.5628},
    "Nevada":     {"registration":12680, "turnout":0.6959,"lean_avg":0.1353,"lean_lin":0.1741,"lean_sd":0.0181,"hist_avg":0.7208},
    "Placer":     {"registration":8640,  "turnout":0.6520,"lean_avg":0.1224,"lean_lin":0.1462,"lean_sd":0.0113,"hist_avg":0.7079},
    "Stanislaus": {"registration":304987,"turnout":0.5178,"lean_avg":-0.1553,"lean_lin":-0.1540,"lean_sd":0.0123,"hist_avg":0.4357},
    "Tuolumne":   {"registration":36167, "turnout":0.6736,"lean_avg":-0.2092,"lean_lin":-0.2034,"lean_sd":0.0140,"hist_avg":0.3763},
}
WIN_THRESHOLD = 0.50
N_DEFAULT     = 10_000

# ── Session state init ────────────────────────────────────────────────────────
# On first load, try to pull from forecast_params.json; fall back to hardcoded
if "county_df" not in st.session_state:
    loaded = load_forecast_params()
    if loaded:
        init_counties = params_to_county_dict(loaded)
        st.session_state["forecast_params"]    = loaded
        st.session_state["district_turnout_sd_pct"] = round(loaded["turnout_sds"]["district_turnout_sd"]*100, 2)
        st.session_state["county_turnout_sd_pct"]   = round(loaded["turnout_sds"]["county_turnout_sd"]*100,   2)
        st.session_state["state_env_sd_pct"]        = round(loaded["state_environment"]["state_env_sd"]*100,  2)
        st.session_state["model_forecast_env"]      = round(loaded["state_environment"]["predicted_state_env"]*100, 2)
    else:
        init_counties = FALLBACK_COUNTIES
        st.session_state["forecast_params"]    = None
        st.session_state["district_turnout_sd_pct"] = 8.23
        st.session_state["county_turnout_sd_pct"]   = 7.73
        st.session_state["state_env_sd_pct"]        = 6.29
        st.session_state["model_forecast_env"]      = 59.80
    st.session_state["county_df"] = params_to_county_df(init_counties)

for key, default in [
    ("district_turnout_sd_pct", 8.23),
    ("county_turnout_sd_pct",   7.73),
    ("state_env_sd_pct",        6.29),
    ("model_forecast_env",      59.80),
    ("forecast_params",         None),
    ("poll_entries",            []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Live parameters from session state ───────────────────────────────────────
DISTRICT_TURNOUT_SD = st.session_state["district_turnout_sd_pct"] / 100
COUNTY_TURNOUT_SD   = st.session_state["county_turnout_sd_pct"]   / 100
STATE_ENV_SD        = st.session_state["state_env_sd_pct"]         / 100

COUNTIES = {}
for _, row in st.session_state["county_df"].iterrows():
    COUNTIES[row["County"]] = {
        "registration": int(row["Registration"]),
        "turnout":      float(row["Turnout (%)"]) / 100,
        "lean_avg":     float(row["Lean Avg (%)"]) / 100,
        "lean_lin":     float(row["Lean Lin (%)"]) / 100,
        "lean_sd":      float(row["Lean SD (%)"]) / 100,
        "hist_avg":     float(row["Hist Avg (%)"]) / 100,
    }

# ── Polling blend: inverse-variance weighted combination ─────────────────────
def load_polling_blend() -> tuple:
    """
    Load current_polls.csv, run MAE-weighted blend with structural model.
    Returns (blended_env, blended_sd, poll_details, mae_coeffs).
    """
    try:
        if str(DATA_DIR) not in sys.path:
            sys.path.insert(0, str(DATA_DIR))
        import importlib
        import polling_model as pm
        importlib.reload(pm)

        curr_path = DATA_DIR / "current_polls.csv"
        if not curr_path.exists():
            return None, None, [], None

        df_hist = pd.read_csv(DATA_DIR / "historical_polls.csv")
        coeffs  = pm.fit_mae_regression(df_hist)
        df_curr = pd.read_csv(curr_path)

        fp = st.session_state.get("forecast_params")
        if fp:
            model_env = fp["state_environment"]["predicted_state_env"]
            model_sd  = fp["state_environment"]["state_env_sd"]
        else:
            model_env = st.session_state.get("model_forecast_env", 59.80) / 100
            model_sd  = st.session_state.get("state_env_sd_pct", 6.29) / 100

        blended_env, blended_sd, details = pm.blend_with_polls(
            model_env, model_sd, df_curr, coeffs
        )
        return blended_env, blended_sd, details, coeffs
    except Exception as e:
        return None, None, [], None
      
# Compute blended state environment using polling model
model_env_val = st.session_state["model_forecast_env"] / 100
_blended_env, _blended_sd, _poll_details, _mae_coeffs = load_polling_blend()
blended_env = _blended_env if _blended_env is not None else model_env_val
blended_sd  = _blended_sd  if _blended_sd  is not None else STATE_ENV_SD

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulation Parameters")
    n_sims = st.number_input("Simulations", min_value=1000, max_value=100_000, value=N_DEFAULT, step=1000)

    st.markdown("**State Environment**")
    use_model_env = st.toggle("Use model forecast", value=True)
    if use_model_env:
        forecast_env = blended_env
        st.caption(f"Model: {model_env_val:.1%} → Blended: {blended_env:.1%} (SD={blended_sd:.1%})")
        STATE_ENV_SD = blended_sd
    else:
        forecast_env = st.number_input("Manual forecast (%)", value=round(blended_env*100,2), step=0.1) / 100

    lean_method = st.radio("County Lean Method", ["Average", "Linear"], horizontal=True)

    st.markdown("---")
    st.markdown("### 🔍 Conditional Filters")
    st.caption("County filters use the threshold below. District win always uses 50%.")
    county_vote_threshold = st.number_input("County Vote Threshold (%)", value=50.0, step=0.5) / 100

    county_filters = {}
    for county in COUNTIES:
        county_filters[county] = st.selectbox(
            county, ["Ignore", "Over threshold", "Under threshold"], key=f"filter_{county}"
        )
    env_lower = st.number_input("State Env Lower (%)", value=0.0,   step=1.0) / 100
    env_upper = st.number_input("State Env Upper (%)", value=100.0, step=1.0) / 100
    run = st.button("▶ Run Simulation", type="primary", width="stretch")

# ── Simulation ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running simulations…")
def run_simulation(n, forecast_env, lean_method_str, seed,
                   district_sd, county_sd, state_sd,
                   counties_frozen, turnout_adj_frozen):
    rng = np.random.default_rng(seed); n = int(n)
    counties    = {name: dict(fields) for name, fields in counties_frozen}
    turnout_adj = dict(turnout_adj_frozen)
    env_error              = rng.normal(0, state_sd, n)
    sim_env                = forecast_env + env_error
    district_turnout_shift = rng.normal(0, district_sd, n)
    lean_key = "lean_avg" if lean_method_str == "Average" else "lean_lin"
    county_shares={}; county_votes={}; county_dem_votes={}; county_turnouts={}
    for name, data in counties.items():
        adj = turnout_adj.get(name, 0.0)
        turnout_sim = np.clip(
            data["turnout"] + adj + district_turnout_shift + rng.normal(0, county_sd, n), 0.01, 1.0)
        lean_error = rng.normal(0, data["lean_sd"], n)
        share      = np.clip(sim_env + data[lean_key] + lean_error, 0.0, 1.0)
        votes      = data["registration"] * turnout_sim
        dem_votes  = votes * share
        county_shares[name]=share; county_votes[name]=votes
        county_dem_votes[name]=dem_votes; county_turnouts[name]=turnout_sim
    total_votes     = sum(county_votes.values())
    total_dem_votes = sum(county_dem_votes.values())
    return {
        "district_share":   total_dem_votes / total_votes,
        "sim_env":          sim_env, "env_error": env_error,
        "county_shares":    county_shares, "county_votes": county_votes,
        "county_dem_votes": county_dem_votes, "county_turnouts": county_turnouts,
    }

def counties_to_frozen(d):
    return tuple((n, tuple(sorted(v.items()))) for n, v in d.items())

if "turnout_adj" not in st.session_state:
    st.session_state["turnout_adj"] = {cn: 0.0 for cn in COUNTIES}

if "sim_results" not in st.session_state or run:
    seed = np.random.randint(0, 2**31)
    st.session_state["sim_results"] = run_simulation(
        n_sims, forecast_env, lean_method, seed,
        DISTRICT_TURNOUT_SD, COUNTY_TURNOUT_SD, STATE_ENV_SD,
        counties_to_frozen(COUNTIES),
        tuple(sorted(st.session_state["turnout_adj"].items()))
    )

res            = st.session_state["sim_results"]
district_share = res["district_share"]
sim_env        = res["sim_env"]
env_error      = res["env_error"]
county_shares  = res["county_shares"]
county_names   = list(COUNTIES.keys())

mask = (sim_env >= env_lower) & (sim_env <= env_upper)
for county, filt in county_filters.items():
    if county not in county_shares: continue
    if filt == "Over threshold":   mask &= county_shares[county] >= county_vote_threshold
    elif filt == "Under threshold": mask &= county_shares[county] <  county_vote_threshold
filtered_share = district_share[mask]; n_filtered = mask.sum()

# ── Helpers ───────────────────────────────────────────────────────────────────
def prob_class(p):
    return "prob-high" if p>=0.60 else ("prob-mid" if p>=0.40 else "prob-low")
def fmt_pct(v, d=1): return f"{v*100:.{d}f}%"

# ══════════════════════════════════════════════════════════════════════════════
st.title("🗳️ SD4 Election Forecast")
st.caption(
    f"Monte Carlo · {int(n_sims):,} simulations · Lean: {lean_method} · "
    f"State env: {fmt_pct(forecast_env)} (SD={fmt_pct(STATE_ENV_SD)}) · "
    f"District win threshold: 50%"
)

tab_dash, tab_turnout, tab_map, tab_model, tab_mechanics, tab_hood = st.tabs([
    "📊 Dashboard", "🎚️ Turnout Explorer", "🗺️ District Map",
    "🔮 Forecast Model", "📐 Model Mechanics", "🔧 Under the Hood"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
    ds=filtered_share if n_filtered>0 else district_share
    wp=float(np.mean(ds>=WIN_THRESHOLD)); n_wins=int(np.sum(ds>=WIN_THRESHOLD))
    col_win,col_stats=st.columns([1,3])
    with col_win:
        st.markdown(f'<div class="win-hero"><div class="wlabel">Win Probability</div><div class="wvalue">{fmt_pct(wp)}</div><div class="wsub">{n_wins:,} of {n_filtered:,} simulations</div></div>', unsafe_allow_html=True)
    with col_stats:
        metrics=[("Mean Share",fmt_pct(np.mean(ds))),("Median Share",fmt_pct(np.median(ds))),
                 ("Std Deviation",fmt_pct(np.std(ds))),("5th Pct",fmt_pct(np.percentile(ds,5))),
                 ("95th Pct",fmt_pct(np.percentile(ds,95)))]
        cols=st.columns(5)
        for col,(lbl,val) in zip(cols,metrics):
            with col:
                st.markdown(f'<div class="stat-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Result Distribution</div>', unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(10,3)); fig.patch.set_facecolor("#f7f7f5"); ax.set_facecolor("#f7f7f5")
    ax.hist(ds*100,bins=60,color="#1a6b3c",alpha=0.75,edgecolor="none")
    ax.axvline(50,color="#b91c1c",linewidth=1.5,linestyle="--",label="Win threshold (50%)")
    ax.axvline(np.mean(ds)*100,color="#0f3d24",linewidth=1.5,linestyle="-",label=f"Mean {fmt_pct(np.mean(ds))}")
    ax.set_xlabel("Dem Vote Share (%)",fontsize=9); ax.set_ylabel("Simulations",fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter()); ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(labelsize=8); ax.legend(fontsize=8,framealpha=0); plt.tight_layout()
    st.pyplot(fig,width="stretch"); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Win Probability by Vote Threshold</div>', unsafe_allow_html=True)
    thresholds=[0.40,0.45,0.50]
    county_headers="".join(f"<th>{cn[:6]}</th>" for cn in county_names)
    rows_html=""
    for thr in thresholds:
        dwp=np.mean(district_share>=thr)
        row=f'<tr><td>&gt;{fmt_pct(thr,0)}</td><td class="{prob_class(dwp)}">{fmt_pct(dwp)}</td>'
        for cn in county_names:
            cwp=np.mean(county_shares[cn]>=thr)
            row+=f'<td class="{prob_class(cwp)}">{fmt_pct(cwp)}</td>'
        rows_html+=row+"</tr>"
    st.markdown(f'<table class="styled-table"><thead><tr><th>Threshold</th><th>District</th>{county_headers}</tr></thead><tbody>{rows_html}</tbody></table><p style="font-size:0.72rem;color:#999;margin-top:0.4rem">District column uses row threshold (not fixed 50%).</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Statistics</div>', unsafe_allow_html=True)
    crow=""
    for cn in county_names:
        d=COUNTIES[cn]; s=county_shares[cn]; ms=np.mean(s); md=np.median(s); h=d["hist_avg"]; r=ms-h
        crow+=f'<tr><td>{cn}</td><td>{fmt_pct(ms)}</td><td>{fmt_pct(md)}</td><td>{fmt_pct(h)}</td><td>{"▲" if r>=0 else "▼"} {fmt_pct(abs(r))}</td><td>{fmt_pct(d["lean_sd"])}</td></tr>'
    st.markdown(f'<table class="styled-table"><thead><tr><th>County</th><th>Mean Share</th><th>Median Share</th><th>Hist. Avg</th><th>Residual</th><th>Lean SD</th></tr></thead><tbody>{crow}</tbody></table>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Win Probability by State Environment</div>', unsafe_allow_html=True)
    bands=[("≤ 0",-np.inf,0.00),("+0 to +5",0.00,0.05),("+5 to +8",0.05,0.08),("+8 to +10",0.08,0.10),("+10 to +12",0.10,0.12),("> +12",0.12,np.inf)]
    erow=""
    for lbl,lo,hi in bands:
        bm=(env_error>=lo)&(env_error<hi); nb=bm.sum()
        nw=int(np.sum(district_share[bm]>=WIN_THRESHOLD)) if nb>0 else 0
        wb=nw/nb if nb>0 else 0.0
        erow+=f"<tr><td>{lbl}</td><td>{nw:,}</td><td>{nb:,}</td><td class='{prob_class(wb)}'>{fmt_pct(wb)}</td></tr>"
    st.markdown(f'<table class="styled-table"><thead><tr><th>Env vs Forecast</th><th>Wins</th><th>Sims</th><th>Win Prob</th></tr></thead><tbody>{erow}</tbody></table>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Path to Victory</div>', unsafe_allow_html=True)
    wm=district_share>=WIN_THRESHOLD; nwt=wm.sum()
    prow=""
    for cn in county_names:
        cw=county_shares[cn]>=WIN_THRESHOLD; pcw=(cw&wm).sum()/nwt if nwt>0 else 0.0
        corr=np.corrcoef(county_shares[cn],sim_env)[0,1]
        prow+=f"<tr><td>{cn}</td><td>{fmt_pct(pcw)}</td><td>{corr:.4f}</td></tr>"
    cde=np.corrcoef(district_share,env_error)[0,1]; cds=np.corrcoef(district_share,sim_env)[0,1]
    col_ptv,col_corr=st.columns([2,1])
    with col_ptv:
        st.markdown(f'<table class="styled-table"><thead><tr><th>County</th><th>P(County Won | District Won)</th><th>Corr vs Env</th></tr></thead><tbody>{prow}</tbody></table>', unsafe_allow_html=True)
    with col_corr:
        st.markdown(f'<div class="stat-card"><div class="label">District Share vs Env Error</div><div class="value" style="font-size:1.3rem">{cde:.4f}</div></div><div class="stat-card"><div class="label">District Share vs Environment</div><div class="value" style="font-size:1.3rem">{cds:.4f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Turnout Analysis</div>', unsafe_allow_html=True)
    fig2,axes=plt.subplots(3,5,figsize=(14,6)); fig2.patch.set_facecolor("#f7f7f5"); af=axes.flatten()
    for i,cn in enumerate(county_names):
        st2=res["county_votes"][cn]/COUNTIES[cn]["registration"]; ax=af[i]; ax.set_facecolor("#f7f7f5")
        ax.hist(st2*100,bins=30,color="#1a6b3c",alpha=0.7,edgecolor="none")
        ax.axvline(COUNTIES[cn]["turnout"]*100,color="#b91c1c",linewidth=1.2,linestyle="--")
        ax.set_title(cn,fontsize=7,fontweight="600"); ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter()); ax.spines[["top","right","left"]].set_visible(False)
    for j in range(len(county_names),len(af)): af[j].set_visible(False)
    plt.suptitle("Simulated Turnout by County  (red = forecast)",fontsize=8,color="#555")
    plt.tight_layout(); st.pyplot(fig2,width="stretch"); plt.close()
    st.markdown("---")
    st.caption(f"Filtered: {n_filtered:,} of {int(n_sims):,} · Env range: {fmt_pct(env_lower)}–{fmt_pct(env_upper)} · Filters active: {sum(1 for v in county_filters.values() if v!='Ignore')}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TURNOUT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_turnout:
    st.markdown("## Turnout Explorer")
    st.markdown("Adjust county turnout relative to forecast. Hit **▶ Run Simulation** to apply.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Turnout Adjustments (vs forecast)</div>', unsafe_allow_html=True)
    adj_cols=st.columns(3); new_adj={}
    for i,cn in enumerate(county_names):
        with adj_cols[i%3]:
            base=COUNTIES[cn]["turnout"]*100; cur=st.session_state["turnout_adj"].get(cn,0.0)
            adj_pct=st.slider(cn,min_value=-10.0,max_value=10.0,value=float(cur*100),step=0.5,
                              format="%.1f%%",help=f"Forecast: {base:.1f}% → Adjusted: {base+cur*100:.1f}%",
                              key=f"ts_{cn}")
            new_adj[cn]=adj_pct/100
    st.session_state["turnout_adj"]=new_adj
    if st.button("↩ Reset all turnout adjustments"):
        st.session_state["turnout_adj"]={cn:0.0 for cn in COUNTIES}; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Projected Impact</div>', unsafe_allow_html=True)
    st.caption("Deterministic estimate of share change. Re-run simulation for updated win probability.")
    irow=""; tbv=0; tav=0; tbd=0; tad=0
    for cn in county_names:
        d=COUNTIES[cn]; adj=new_adj.get(cn,0.0); ms=float(np.mean(county_shares[cn]))
        bt=d["turnout"]; at=np.clip(bt+adj,0.01,1.0); reg=d["registration"]
        bv=reg*bt; av=reg*at; bd=bv*ms; ad=av*ms; dd=ad-bd
        tbv+=bv; tav+=av; tbd+=bd; tad+=ad
        irow+=(f"<tr><td>{cn}</td><td>{fmt_pct(bt)}</td><td>{fmt_pct(at)}</td>"
               f"<td>{bv:,.0f}</td><td>{av:,.0f}</td><td>{fmt_pct(ms)}</td>"
               f"<td>{'▲' if dd>=0 else '▼'} {abs(dd):,.0f}</td></tr>")
    bs=tbd/tbv if tbv>0 else 0; as_=tad/tav if tav>0 else 0; ds=as_-bs
    st.markdown(f'<table class="styled-table"><thead><tr><th>County</th><th>Base Turnout</th><th>Adj Turnout</th><th>Base Votes</th><th>Adj Votes</th><th>Mean Dem Share</th><th>Δ Dem Votes</th></tr></thead><tbody>{irow}</tbody></table>', unsafe_allow_html=True)
    ic1,ic2,ic3=st.columns(3)
    for col,(lbl,val) in zip([ic1,ic2,ic3],[("Base District Share",bs),("Adjusted District Share",as_),("Share Change",ds)]):
        with col:
            arrow="▲" if val>=0 or lbl!="Share Change" else "▼"
            st.markdown(f'<div class="stat-card"><div class="label">{lbl}</div><div class="value">{("▲" if ds>=0 else "▼") + " " if lbl=="Share Change" else ""}{fmt_pct(abs(val) if lbl=="Share Change" else val)}</div></div>', unsafe_allow_html=True)
    st.info("💡 Hit **▶ Run Simulation** in the sidebar to run the full Monte Carlo with these turnout adjustments.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("## District Map — SD4")
    map_col,info_col=st.columns([2,1])
    with map_col:
        metric_options={
            "Mean Dem Vote Share":    lambda cn: float(np.mean(county_shares[cn])),
            "Win Probability (>50%)": lambda cn: float(np.mean(county_shares[cn]>=WIN_THRESHOLD)),
            "Win Probability (>45%)": lambda cn: float(np.mean(county_shares[cn]>=0.45)),
            "Win Probability (>40%)": lambda cn: float(np.mean(county_shares[cn]>=0.40)),
            "Median Dem Vote Share":  lambda cn: float(np.median(county_shares[cn])),
            "Std Deviation":          lambda cn: float(np.std(county_shares[cn])),
            "Historical Average":     lambda cn: COUNTIES[cn]["hist_avg"],
        }
        sel_metric=st.selectbox("Choropleth metric",list(metric_options.keys()))
        fn=metric_options[sel_metric]
        cv={cn:fn(cn) for cn in county_names}; vmin=min(cv.values()); vmax=max(cv.values())
        sel_county=st.session_state.get("selected_county_map",None)
        county_polys=load_county_geojson()
        cmap=plt.colormaps["bwr_r"]
        fig_map,ax_map=plt.subplots(figsize=(9,8)); fig_map.patch.set_facecolor("#f0f0ec"); ax_map.set_facecolor("#c8dff0")
        sx=[p[0] for p in SD4_BOUNDARY]; sy=[p[1] for p in SD4_BOUNDARY]
        ax_map.fill(sx,sy,color="#e8e4dc",alpha=1.0,zorder=0); ax_map.plot(sx,sy,color="#222",linewidth=2.5,zorder=3)
        if not county_polys:
            ax_map.text(0.5,0.5,"County boundaries unavailable.\nCheck internet connection.",transform=ax_map.transAxes,ha="center",va="center",fontsize=10,color="#666")
        else:
            texts=[]
            for cn in county_names:
                if cn not in county_polys: continue
                ring=county_polys[cn]; val=cv.get(cn,0.5)
                t=(val-vmin)/(vmax-vmin) if vmax>vmin else 0.5
                color=cmap(t); xs=[float(p[0]) for p in ring]; ys=[float(p[1]) for p in ring]
                is_sel=(cn==sel_county)
                ax_map.fill(xs,ys,color=color,alpha=0.88,zorder=1,linewidth=2.5 if is_sel else 0.6,edgecolor="#FFD700" if is_sel else "#ffffff")
                if is_sel: ax_map.plot(xs,ys,color="#FFD700",linewidth=2.5,zorder=4)
                cx,cy=COUNTY_CENTROIDS.get(cn,(np.mean(xs),np.mean(ys)))
                vl=f"{val*100:.1f}%"
                texts.append(ax_map.text(cx,cy,f"{cn}\n{vl}",ha="center",va="center",fontsize=5.5,fontweight="bold",color="white",zorder=5,path_effects=[pe.withStroke(linewidth=1.8,foreground="#222")],multialignment="center"))
            try:
                from adjustText import adjust_text
                adjust_text(texts,ax=ax_map,expand=(1.3,1.5),force_text=(0.5,0.8),force_static=(0.3,0.5),arrowprops=dict(arrowstyle="-",color="#444",lw=0.6))
            except ImportError:
                pass
        ax_map.set_xlim(-122.0,-115.4); ax_map.set_ylim(35.6,39.6); ax_map.set_aspect("equal"); ax_map.axis("off")
        ax_map.set_title(f"CA Senate District 4 — {sel_metric}",fontsize=9,pad=8,color="#333")
        sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=vmin,vmax=vmax)); sm.set_array([])
        cbar=fig_map.colorbar(sm,ax=ax_map,fraction=0.025,pad=0.02); cbar.ax.tick_params(labelsize=7)
        if any(k in sel_metric for k in ("Probability","Share","Average")): cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        plt.tight_layout(); st.pyplot(fig_map,width="stretch"); plt.close()

    with info_col:
        st.markdown("### County Detail")
        sel=st.selectbox("Select county",["— select —"]+county_names,key="county_selector")
        st.session_state["selected_county_map"]=sel if sel!="— select —" else None
        if sel and sel!="— select —":
            cn=sel; d=COUNTIES[cn]; shr=county_shares[cn]; wm2=district_share>=WIN_THRESHOLD
            ms=float(np.mean(shr)); med=float(np.median(shr)); sd=float(np.std(shr))
            p5=float(np.percentile(shr,5)); p95=float(np.percentile(shr,95))
            wp40=float(np.mean(shr>=0.40)); wp45=float(np.mean(shr>=0.45)); wp50=float(np.mean(shr>=0.50))
            h=d["hist_avg"]; r2=ms-h; cw2=shr>=WIN_THRESHOLD
            pcw2=float((cw2&wm2).sum()/wm2.sum()) if wm2.sum()>0 else 0.0
            corr2=float(np.corrcoef(shr,sim_env)[0,1]); adj2=st.session_state["turnout_adj"].get(cn,0.0)
            st.markdown(f"""<table class="styled-table" style="font-size:0.78rem;"><tbody>
              <tr><td><b>Registration</b></td><td>{d['registration']:,}</td></tr>
              <tr><td><b>Forecast turnout</b></td><td>{fmt_pct(d['turnout'])}</td></tr>
              <tr><td><b>Turnout adjustment</b></td><td>{'+' if adj2>=0 else ''}{fmt_pct(adj2)}</td></tr>
              <tr><td colspan="2" style="padding-top:0.6rem;font-weight:600;color:#1a6b3c">Simulated Vote Share</td></tr>
              <tr><td>Mean</td><td>{fmt_pct(ms)}</td></tr><tr><td>Median</td><td>{fmt_pct(med)}</td></tr>
              <tr><td>Std Dev</td><td>{fmt_pct(sd)}</td></tr>
              <tr><td>5th pct</td><td>{fmt_pct(p5)}</td></tr><tr><td>95th pct</td><td>{fmt_pct(p95)}</td></tr>
              <tr><td colspan="2" style="padding-top:0.6rem;font-weight:600;color:#1a6b3c">Win Probabilities</td></tr>
              <tr><td>P(share &gt; 40%)</td><td>{fmt_pct(wp40)}</td></tr>
              <tr><td>P(share &gt; 45%)</td><td>{fmt_pct(wp45)}</td></tr>
              <tr><td>P(share &gt; 50%)</td><td>{fmt_pct(wp50)}</td></tr>
              <tr><td colspan="2" style="padding-top:0.6rem;font-weight:600;color:#1a6b3c">Historical</td></tr>
              <tr><td>Historical avg</td><td>{fmt_pct(h)}</td></tr>
              <tr><td>Residual</td><td>{'▲' if r2>=0 else '▼'} {fmt_pct(abs(r2))}</td></tr>
              <tr><td>Lean SD</td><td>{fmt_pct(d['lean_sd'])}</td></tr>
              <tr><td colspan="2" style="padding-top:0.6rem;font-weight:600;color:#1a6b3c">Path to Victory</td></tr>
              <tr><td>P(won | district won)</td><td>{fmt_pct(pcw2)}</td></tr>
              <tr><td>Corr w/ state env</td><td>{corr2:.4f}</td></tr>
            </tbody></table>""", unsafe_allow_html=True)
            fig_mini,ax_mini=plt.subplots(figsize=(4,2)); fig_mini.patch.set_facecolor("#f7f7f5"); ax_mini.set_facecolor("#f7f7f5")
            ax_mini.hist(shr*100,bins=40,color="#1a6b3c",alpha=0.7,edgecolor="none")
            ax_mini.axvline(50,color="#b91c1c",linewidth=1.2,linestyle="--")
            ax_mini.set_title(f"{cn} vote share distribution",fontsize=7)
            ax_mini.xaxis.set_major_formatter(mtick.PercentFormatter()); ax_mini.spines[["top","right","left"]].set_visible(False)
            ax_mini.tick_params(labelsize=6); plt.tight_layout(); st.pyplot(fig_mini,width="stretch"); plt.close()
        else:
            st.info("Select a county to see detailed statistics.")
            st.markdown(f'<div class="stat-card"><div class="label">District Win Probability</div><div class="value">{fmt_pct(float(np.mean(district_share>=WIN_THRESHOLD)))}</div><div class="sub">across all {int(n_sims):,} simulations</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORECAST MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("## Forecast Model")
    st.markdown(
        "Update the underlying forecast by changing election context inputs and re-running the model. "
        "This re-runs all four regressions and updates county parameters, turnout forecasts, "
        "state environment, and SDs. After updating, hit **▶ Run Simulation** to apply."
    )

    fp = st.session_state.get("forecast_params")
    if fp:
        ctx = fp.get("context", {})
        st.success(
            f"Model last run: **{ctx.get('year','')} {'General' if ctx.get('general') else 'Primary'}** · "
            f"Inflation: {ctx.get('inflation','?')}% · Approval: {ctx.get('approval',0):.0%} · "
            f"Predicted env: {fp['state_environment']['predicted_state_env']:.1%}"
        )
    else:
        st.warning("No forecast_params.json found. Using hardcoded defaults. Run the model below to generate parameters.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Election Context</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        ctx_year    = st.number_input("Election Year", value=2026, step=2)
        ctx_general = st.toggle("General Election", value=False)
        ctx_pres    = st.toggle("Presidential Cycle", value=False)
    with c2:
        ctx_inflation = st.number_input("Inflation (CPI YoY %)", value=float(fp["context"]["inflation"]) if fp else 3.8, step=0.1, format="%.1f")
        ctx_approval  = st.number_input("Presidential Approval (0–1)", value=float(fp["context"]["approval"]) if fp else 0.39, step=0.01, format="%.2f", min_value=0.0, max_value=1.0)
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_model = st.button("🔄 Run Forecast Model", type="primary", width="stretch")
        st.caption("Updates all parameters from historical CSVs using the context above.")

    if run_model:
        ctx_input = {
            "year": int(ctx_year), "general": bool(ctx_general),
            "presidential": bool(ctx_pres), "inflation": float(ctx_inflation),
            "approval": float(ctx_approval),
        }
        with st.spinner("Running forecast model…"):
            new_params = run_forecast_model(ctx_input)
        if new_params:
            new_counties = params_to_county_dict(new_params)
            st.session_state["county_df"]               = params_to_county_df(new_counties)
            st.session_state["forecast_params"]          = new_params
            st.session_state["district_turnout_sd_pct"] = round(new_params["turnout_sds"]["district_turnout_sd"]*100, 2)
            st.session_state["county_turnout_sd_pct"]   = round(new_params["turnout_sds"]["county_turnout_sd"]*100,   2)
            st.session_state["state_env_sd_pct"]        = round(new_params["state_environment"]["state_env_sd"]*100,  2)
            st.session_state["model_forecast_env"]      = round(new_params["state_environment"]["predicted_state_env"]*100, 2)
            st.success("Model updated! Hit ▶ Run Simulation in the sidebar to apply.")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

   # ── Polling inputs ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Polling — State Environment</div>', unsafe_allow_html=True)
    st.markdown(
        "Polls are loaded from **current_polls.csv** in the same folder as the app. "
        "Add rows to that file (source, end_date, election_date, days_out, cycle, race, "
        "election, sample_size, type, rv, lv, dem, rep) and click **Reload Polls** below. "
        "Weights are computed automatically using the MAE regression from historical polling accuracy."
    )

    col_reload, col_csv, _ = st.columns([1, 1, 3])
    with col_reload:
        if st.button("🔄 Reload Polls"):
            st.rerun()
    with col_csv:
        curr_path2 = DATA_DIR / "current_polls.csv"
        if curr_path2.exists():
            with open(curr_path2) as f2:
                st.download_button("⬇ Download current_polls.csv", f2.read(),
                                   file_name="current_polls.csv", mime="text/csv")

    details = _poll_details
    if details:
        poll_mean = sum(d["dem"] for d in details) / len(details)
        st.markdown(f"""
        <div class="poll-blend">
          <b>Blended state environment: {blended_env:.2%}</b> (SD={blended_sd:.2%})<br>
          <span style="font-size:0.82rem;color:#555">
            Structural model: {model_env_val:.2%} · Simple poll average: {poll_mean:.2%}
             · {len(details)} poll{"s" if len(details)!=1 else ""} loaded
          </span>
        </div>
        """, unsafe_allow_html=True)

        prows = ""
        for d in details:
            poll_type = "LV" if d["lv"] else ("RV" if d["rv"] else "—")
            partisan_badge = ' <span style="color:#d97706;font-size:0.7rem">[partisan]</span>' if d["partisan"] else ""
            prows += (
                f"<tr>"
                f"<td>{d['source'][:40]}{partisan_badge}</td>"
                f"<td>{fmt_pct(d['dem'])}</td>"
                f"<td>{poll_type}</td>"
                f"<td>{d['days_out']:.0f}</td>"
                f"<td>{fmt_pct(d['pred_mae'])}</td>"
                f"<td>{d['rel_weight']:.3f}</td>"
                f"</tr>"
            )
        st.markdown(
            f'<table class="styled-table"><thead><tr>'
            f'<th>Poll</th><th>Dem Share</th><th>Type</th>'
            f'<th>Days Out</th><th>Pred MAE</th><th>Rel Weight</th>'
            f'</tr></thead><tbody>{prows}</tbody></table>',
            unsafe_allow_html=True
        )
        st.caption(
            "Rel Weight = each poll's share of total poll precision. "
            "Partisan polls (marked D/R) receive a 50% weight discount."
        )
    else:
        st.info(
            "No polls loaded. Add rows to current_polls.csv and click Reload Polls. "
            "The simulation currently uses the structural model only."
        )

    # ── Data files status ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Data File Status</div>', unsafe_allow_html=True)
    files = [
        ("historical_turnout.csv",        "County turnout by election 2012–present"),
        ("historical_voting.csv",         "County Dem shares and leans by election"),
        ("state_environment_history.csv", "Statewide Dem share with model features"),
        ("registration_history.csv",      "Registration history and 2026 forecast"),
        ("partial_county_fractions.csv",  "SD4 fractions for partial counties"),
        ("forecast_params.json",          "Current model output (auto-updated above)"),
    ]
    frows=""
    for fname,desc in files:
        p2=DATA_DIR/fname
        if p2.exists():
            import os; mtime=pd.Timestamp(os.path.getmtime(p2),unit="s").strftime("%Y-%m-%d %H:%M")
            frows+=f'<tr><td>✅ {fname}</td><td>{desc}</td><td>{mtime}</td></tr>'
        else:
            frows+=f'<tr><td>❌ {fname}</td><td>{desc}</td><td>Not found</td></tr>'
    st.markdown(f'<table class="styled-table"><thead><tr><th>File</th><th>Contents</th><th>Last Modified</th></tr></thead><tbody>{frows}</tbody></table>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL MECHANICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_mechanics:
    st.markdown("## Model Mechanics")
    fp2 = st.session_state.get("forecast_params")
    if not fp2:
        st.info("Run the Forecast Model (tab above) to populate this tab.")
        st.stop()

    ctx2   = fp2["context"]
    state2 = fp2["state_environment"]
    sds2   = fp2["turnout_sds"]
    tcoef  = fp2.get("turnout_coefficients", {})

    # ── State environment model ───────────────────────────────────────────────
    st.markdown('<div class="section-label">State Environment Model</div>', unsafe_allow_html=True)
    st.markdown(
        f"OLS regression on {state2['n_obs']} historical CA statewide elections. "
        f"Features: presidential cycle dummy, inflation (CPI YoY %), presidential approval (0–1)."
    )
    sc = state2["coefficients"]
    coef_cols = st.columns(4)
    for col, (name, val) in zip(coef_cols, sc.items()):
        with col:
            st.markdown(f'<div class="stat-card"><div class="label">{name}</div><div class="value" style="font-size:1.1rem">{val:+.4f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Back-test chart: predicted vs actual
    try:
        df_se = pd.read_csv(DATA_DIR / "state_environment_history.csv")
        coefs = sc
        df_se["predicted"] = (coefs["intercept"]
            + coefs["presidential"] * df_se["presidential"]
            + coefs["inflation"]    * df_se["inflation"]
            + coefs["approval"]     * df_se["approval"])
        df_se["residual"] = df_se["dem_share"] - df_se["predicted"]

        fig_bt, axes_bt = plt.subplots(1, 2, figsize=(12, 3.5))
        fig_bt.patch.set_facecolor("#f7f7f5")
        ax1, ax2 = axes_bt

        ax1.set_facecolor("#f7f7f5")
        ax1.scatter(df_se["predicted"]*100, df_se["dem_share"]*100, color="#1a6b3c", alpha=0.7, s=40)
        mn = min(df_se["predicted"].min(), df_se["dem_share"].min())*100 - 1
        mx = max(df_se["predicted"].max(), df_se["dem_share"].max())*100 + 1
        ax1.plot([mn,mx],[mn,mx],color="#ccc",linewidth=1,linestyle="--")
        ax1.set_xlabel("Predicted Dem Share (%)",fontsize=8); ax1.set_ylabel("Actual Dem Share (%)",fontsize=8)
        ax1.xaxis.set_major_formatter(mtick.PercentFormatter()); ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_title("State Environment: Predicted vs Actual",fontsize=8,fontweight="600")
        ax1.spines[["top","right"]].set_visible(False); ax1.tick_params(labelsize=7)

        ax2.set_facecolor("#f7f7f5")
        ax2.bar(range(len(df_se)), df_se["residual"]*100, color=["#1a6b3c" if r>=0 else "#b91c1c" for r in df_se["residual"]], alpha=0.7)
        ax2.axhline(0, color="#ccc", linewidth=0.8)
        ax2.set_xticks(range(len(df_se))); ax2.set_xticklabels([e[:12] for e in df_se["election"]], rotation=45, ha="right", fontsize=5.5)
        ax2.set_ylabel("Residual (%)",fontsize=8); ax2.set_title("Residuals by Election",fontsize=8,fontweight="600")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter()); ax2.spines[["top","right","left"]].set_visible(False); ax2.tick_params(labelsize=7)
        plt.tight_layout(); st.pyplot(fig_bt,width="stretch"); plt.close()

        rmse = float(np.sqrt((df_se["residual"]**2).mean()))
        st.caption(f"Model RMSE: {rmse:.3f} · Residual SD (used in simulation): {state2['state_env_sd']:.4f}")
    except Exception as e:
        st.caption(f"Could not load state environment history for chart: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Turnout model ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Turnout Model</div>', unsafe_allow_html=True)
    st.markdown("Panel OLS with county fixed effects. Features: presidential cycle, general/primary dummy, previous same-type election turnout.")
    if tcoef:
        key_coefs = {k:v for k,v in tcoef.items() if k in ("intercept","presidential","general","prev_turnout")}
        tc_cols = st.columns(4)
        for col,(n,v) in zip(tc_cols, key_coefs.items()):
            with col:
                st.markdown(f'<div class="stat-card"><div class="label">{n}</div><div class="value" style="font-size:1.1rem">{v:+.4f}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

       # Predicted turnout chart (more intuitive than raw fixed effects)
    try:
        df_t2 = pd.read_csv(DATA_DIR / "historical_turnout.csv")
        df_t2 = df_t2[df_t2.votes_cast.notna()].copy()

        fig_to, ax_to = plt.subplots(figsize=(11, 4))
        fig_to.patch.set_facecolor("#f7f7f5"); ax_to.set_facecolor("#f7f7f5")

        x = np.arange(len(county_names)); w = 0.35
        is_gen = fp2["context"]["general"]
        if is_gen:
            hist_means = [
                df_t2[(df_t2.county==cn) & (df_t2.general==1)]["turnout_rate"].mean()
                for cn in county_names
            ]
            hist_label = "Historical mean (generals)"
        else:
            hist_means = [
                df_t2[(df_t2.county==cn) & (df_t2.primary==1)]["turnout_rate"].mean()
                for cn in county_names
            ]
            hist_label = "Historical mean (primaries)"

        pred_turns = [fp2["counties"][cn]["turnout"] for cn in county_names]

        ax_to.bar(x - w/2, [v*100 for v in hist_means], w,
                  label=hist_label, color="#888", alpha=0.6)
        ax_to.bar(x + w/2, [v*100 for v in pred_turns], w,
                  label="Model forecast", color="#1a6b3c", alpha=0.8)
        ax_to.set_xticks(x)
        ax_to.set_xticklabels(county_names, rotation=40, ha="right", fontsize=7)
        ax_to.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax_to.set_ylabel("Turnout (%)", fontsize=8)
        ax_to.set_title("County Turnout: Historical Mean vs Model Forecast", fontsize=8, fontweight="600")
        ax_to.legend(fontsize=8, framealpha=0)
        ax_to.spines[["top","right","left"]].set_visible(False)
        ax_to.tick_params(labelsize=7)
        plt.tight_layout(); st.pyplot(fig_to, width="stretch"); plt.close()
    except Exception as e:
        st.caption(f"Could not render turnout chart: {e}")

    sd_c1, sd_c2 = st.columns(2)
    with sd_c1:
        st.markdown(f'<div class="stat-card"><div class="label">District Turnout SD</div><div class="value">{fmt_pct(sds2["district_turnout_sd"])}</div><div class="sub">Residual SD from turnout regression</div></div>', unsafe_allow_html=True)
    with sd_c2:
        st.markdown(f'<div class="stat-card"><div class="label">County Turnout SD</div><div class="value">{fmt_pct(sds2["county_turnout_sd"])}</div><div class="sub">Mean per-county residual SD</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Forecast Summary</div>', unsafe_allow_html=True)
    st.caption("All figures from the current forecast model run.")
    try:
        is_gen = fp2["context"]["general"]
        lean_key = "lean_avg" if lean_method == "Average" else "lean_lin"
        frows = ""
        total_reg = 0; total_votes = 0; total_dem = 0
        for cn in county_names:
            cd = fp2["counties"][cn]
            reg  = cd["registration"]
            to   = cd["turnout"]
            lean = cd[lean_key]
            env  = fp2["state_environment"]["predicted_state_env"]
            share = env + lean
            votes = reg * to
            dem   = votes * share
            total_reg   += reg
            total_votes += votes
            total_dem   += dem
            frows += (
                f"<tr><td>{cn}</td>"
                f"<td>{reg:,}</td>"
                f"<td>{to*100:.1f}%</td>"
                f"<td>{votes:,.0f}</td>"
                f"<td>{env*100:.1f}%</td>"
                f"<td>{lean*100:+.1f}%</td>"
                f"<td>{share*100:.1f}%</td>"
                f"<td>{dem:,.0f}</td></tr>"
            )
        dist_share = total_dem / total_votes if total_votes > 0 else 0
        frows += (
            f"<tr style='font-weight:600;background:#f0f7f3'>"
            f"<td>District Total</td>"
            f"<td>{total_reg:,}</td>"
            f"<td>—</td>"
            f"<td>{total_votes:,.0f}</td>"
            f"<td>—</td><td>—</td>"
            f"<td>{dist_share*100:.1f}%</td>"
            f"<td>{total_dem:,.0f}</td></tr>"
        )
        st.markdown(
            f'<table class="styled-table">'
            f'<thead><tr><th>County</th><th>Registration</th><th>Turnout</th>'
            f'<th>Est. Votes</th><th>State Env</th><th>Lean ({lean_method})</th>'
            f'<th>Forecast Share</th><th>Est. Dem Votes</th></tr></thead>'
            f'<tbody>{frows}</tbody></table>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.caption(f"Could not render forecast table: {e}")

    # ── County leans ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">County Leans — Historical vs Forecast</div>', unsafe_allow_html=True)
    try:
        df_v2 = pd.read_csv(DATA_DIR / "historical_voting.csv")
        fig_leans, ax_l = plt.subplots(figsize=(11,4))
        fig_leans.patch.set_facecolor("#f7f7f5"); ax_l.set_facecolor("#f7f7f5")
        x=np.arange(len(county_names)); w=0.25
        avg_leans  = [COUNTIES[cn]["lean_avg"]*100  for cn in county_names]
        lin_leans  = [COUNTIES[cn]["lean_lin"]*100  for cn in county_names]
        hist_sds   = [COUNTIES[cn]["lean_sd"]*100   for cn in county_names]
        ax_l.bar(x-w/2, avg_leans, w, label="Weighted avg lean", color="#1a6b3c", alpha=0.8)
        ax_l.bar(x+w/2, lin_leans, w, label="Linear trend lean",  color="#0f3d24", alpha=0.5)
        ax_l.errorbar(x-w/2, avg_leans, yerr=[s*1.96 for s in hist_sds], fmt="none", ecolor="#555", elinewidth=1, capsize=3)
        ax_l.axhline(0, color="#ccc", linewidth=0.8)
        ax_l.set_xticks(x); ax_l.set_xticklabels(county_names, rotation=40, ha="right", fontsize=7)
        ax_l.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax_l.set_ylabel("Lean vs State Env (%)", fontsize=8)
        ax_l.set_title("County Leans — Weighted Average vs Linear Trend (error bars = 95% CI)", fontsize=8, fontweight="600")
        ax_l.legend(fontsize=8, framealpha=0); ax_l.spines[["top","right","left"]].set_visible(False); ax_l.tick_params(labelsize=7)
        plt.tight_layout(); st.pyplot(fig_leans, width="stretch"); plt.close()
    except Exception as e:
        st.caption(f"Could not render lean chart: {e}")
# Historical lean table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Leans by Election</div>', unsafe_allow_html=True)
    st.caption("Each cell shows county Dem share minus statewide Dem share for that election. "
               "Avg lean and linear lean are the two forecast methods.")
    try:
        df_lv = pd.read_csv(DATA_DIR / "historical_voting.csv")
        ELECTION_ORDER = [
            "2022_ss4_pri", "2022_gov_pri", "2022_senate_pri",
            "2022_gov_gen", "2022_senate_gen",
            "2024_senate_pri", "2024_pres_gen", "2024_senate_gen",
        ]
        pivot = df_lv[df_lv.state_dem_share.notna()].pivot_table(
            index="county", columns="election", values="county_lean", aggfunc="first"
        )
        # Reorder columns to election sequence
        cols_present = [e for e in ELECTION_ORDER if e in pivot.columns]
        pivot = pivot[cols_present]

        # Add forecast columns
        pivot["Avg Lean"] = [
            fp2["counties"][cn]["lean_avg"] if cn in fp2["counties"] else np.nan
            for cn in pivot.index
        ]
        pivot["Lin Lean"] = [
            fp2["counties"][cn]["lean_lin"] if cn in fp2["counties"] else np.nan
            for cn in pivot.index
        ]

        # Reorder rows to match county_names order
        pivot = pivot.reindex([cn for cn in county_names if cn in pivot.index])

        # Format as HTML table with colour coding
        def lean_cell(v):
            if pd.isna(v): return "<td>—</td>"
            color = "#1a6b3c" if v >= 0 else "#b91c1c"
            return f'<td style="color:{color};font-weight:500">{v*100:+.1f}%</td>'

        header = "<th>County</th>" + "".join(
            f"<th>{c.replace('_',' ')}</th>" for c in pivot.columns
        )
        rows_lt = ""
        for cn, row in pivot.iterrows():
            is_forecast = lambda c: c in ("Avg Lean", "Lin Lean")
            cells = "".join(
                f'<td style="background:#f0f7f3;font-weight:600;color:{"#1a6b3c" if row[c]>=0 else "#b91c1c"}">{row[c]*100:+.1f}%</td>'
                if is_forecast(c) else lean_cell(row[c])
                for c in pivot.columns
            )
            rows_lt += f"<tr><td>{cn}</td>{cells}</tr>"

        st.markdown(
            f'<table class="styled-table"><thead><tr>{header}</tr></thead>'
            f'<tbody>{rows_lt}</tbody></table>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.caption(f"Could not render lean table: {e}")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Polling blend visualisation ───────────────────────────────────────────
    if _poll_details:
        st.markdown('<div class="section-label">Polling Blend Illustration</div>', unsafe_allow_html=True)
        from scipy.stats import norm as _norm
        x_range   = np.linspace(0.40, 0.75, 400)
        model_pdf = _norm.pdf(x_range, model_env_val, STATE_ENV_SD)
        blend_pdf = _norm.pdf(x_range, blended_env, blended_sd)
        poll_avg  = sum(d["dem"] for d in _poll_details) / len(_poll_details)
        fig_pb, ax_pb = plt.subplots(figsize=(9, 3))
        fig_pb.patch.set_facecolor("#f7f7f5"); ax_pb.set_facecolor("#f7f7f5")
        ax_pb.plot(x_range*100, model_pdf, color="#888", linewidth=1.5, linestyle="--",
                   label=f"Structural model ({model_env_val:.1%}, SD={STATE_ENV_SD:.2%})")
        ax_pb.plot(x_range*100, blend_pdf, color="#1a6b3c", linewidth=2.0,
                   label=f"Blended ({blended_env:.1%}, SD={blended_sd:.2%})")
        ax_pb.axvline(poll_avg*100, color="#d97706", linewidth=1.5, linestyle=":",
                      label=f"Simple poll avg ({poll_avg:.1%})")
        ax_pb.set_xlabel("State Dem Share (%)", fontsize=8); ax_pb.set_ylabel("Density", fontsize=8)
        ax_pb.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax_pb.legend(fontsize=7, framealpha=0)
        ax_pb.spines[["top","right","left"]].set_visible(False); ax_pb.tick_params(labelsize=7)
        ax_pb.set_title("State Environment: Structural Prior vs Polling Blend", fontsize=8, fontweight="600")
        plt.tight_layout(); st.pyplot(fig_pb, width="stretch"); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — UNDER THE HOOD
# ══════════════════════════════════════════════════════════════════════════════
with tab_hood:
    st.markdown("## Under the Hood")
    st.markdown(
        f"County-level Monte Carlo simulation. Each of the **{int(n_sims):,} simulations** "
        "independently draws random values for the state environment, county turnouts, and county "
        "vote shares, then rolls them up to a single district result."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Global Standard Deviations</div>', unsafe_allow_html=True)
    st.caption("Edit and hit ▶ Run Simulation to apply. These are normally set automatically by the forecast model.")
    sd_c1,sd_c2,sd_c3,_=st.columns([1,1,1,2])
    with sd_c1:
        st.number_input("District Turnout SD (%)", min_value=0.0, max_value=20.0, step=0.05,
                        format="%.2f", key="in_dst_sd",
                        value=float(st.session_state["district_turnout_sd_pct"]),
                        on_change=lambda: st.session_state.update({"district_turnout_sd_pct": st.session_state["in_dst_sd"]}))
    with sd_c2:
        st.number_input("County Turnout SD (%)", min_value=0.0, max_value=20.0, step=0.05,
                        format="%.2f", key="in_cty_sd",
                        value=float(st.session_state["county_turnout_sd_pct"]),
                        on_change=lambda: st.session_state.update({"county_turnout_sd_pct": st.session_state["in_cty_sd"]}))
    with sd_c3:
        st.number_input("State Env SD (%)", min_value=0.0, max_value=20.0, step=0.05,
                        format="%.2f", key="in_env_sd",
                        value=float(st.session_state["state_env_sd_pct"]),
                        on_change=lambda: st.session_state.update({"state_env_sd_pct": st.session_state["in_env_sd"]}))
    if st.button("↩ Reset SDs to model values",key="rst_sds"):
        fp3=st.session_state.get("forecast_params")
        if fp3:
            st.session_state["district_turnout_sd_pct"]=round(fp3["turnout_sds"]["district_turnout_sd"]*100,2)
            st.session_state["county_turnout_sd_pct"]  =round(fp3["turnout_sds"]["county_turnout_sd"]*100,2)
            st.session_state["state_env_sd_pct"]       =round(fp3["state_environment"]["state_env_sd"]*100,2)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">County Parameters</div>', unsafe_allow_html=True)
    st.caption("All % fields in plain % (e.g. 49.38). Edit and hit ▶ Run Simulation to apply. "
               "Use the Forecast Model tab to regenerate from data.")

    def apply_county_edits():
        es=st.session_state.get("county_editor",{})
        if not es: return
        df=st.session_state["county_df"].copy()
        for idx_str,changes in es.get("edited_rows",{}).items():
            idx=int(idx_str)
            for col,val in changes.items(): df.at[idx,col]=val
        for row in es.get("added_rows",[]): df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
        di=es.get("deleted_rows",[])
        if di: df=df.drop(index=di).reset_index(drop=True)
        st.session_state["county_df"]=df

    st.data_editor(
        st.session_state["county_df"], width="stretch", hide_index=True, disabled=["County"],
        column_config={
            "County":       st.column_config.TextColumn("County",width="medium"),
            "Registration": st.column_config.NumberColumn("Predicted Registration",min_value=0,step=1,format="%d"),
            "Turnout (%)":  st.column_config.NumberColumn("Turnout (%)", min_value=0.0,max_value=100.0,step=0.01,format="%.2f"),
            "Lean Avg (%)": st.column_config.NumberColumn("Lean Avg (%)",min_value=-100.0,max_value=100.0,step=0.01,format="%.2f"),
            "Lean Lin (%)": st.column_config.NumberColumn("Lean Lin (%)",min_value=-100.0,max_value=100.0,step=0.01,format="%.2f"),
            "Lean SD (%)":  st.column_config.NumberColumn("Lean SD (%)", min_value=0.0,max_value=50.0,step=0.01,format="%.2f"),
            "Hist Avg (%)": st.column_config.NumberColumn("Hist Avg (%)",min_value=0.0,max_value=100.0,step=0.01,format="%.2f"),
        },
        key="county_editor", on_change=apply_county_edits,
    )

    col_rst,_=st.columns([1,3])
    with col_rst:
        if st.button("↩ Reset to model values",key="rst_cty"):
            fp4=st.session_state.get("forecast_params")
            if fp4:
                st.session_state["county_df"]=params_to_county_df(params_to_county_dict(fp4))
            else:
                st.session_state["county_df"]=params_to_county_df(FALLBACK_COUNTIES)
            st.rerun()
