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

# Embed simplified SD4 boundary (RDP-simplified from official GeoJSON)
_SD4_RAW = [(-116.47215,36.44656),(-115.64837,35.80922),(-115.73576,35.8091),(-115.85258,35.83729),(-116.09538,35.9271),(-116.47215,36.14578),(-116.68895,36.28051),(-116.90575,36.44656),(-117.01261,36.51258),(-117.18231,36.60966),(-117.46804,36.78088),(-117.5749,36.87796),(-117.79171,36.97503),(-118.00851,37.07211),(-118.07897,37.18629),(-118.22537,37.28337),(-118.30943,37.45459),(-118.43948,37.55167),(-118.55473,37.68085),(-118.64519,37.87262),(-118.69888,37.98134),(-118.80574,38.11053),(-118.91161,38.22471),(-119.00207,38.3389),(-119.09253,38.45308),(-119.18299,38.58227),(-119.32321,38.67935),(-119.41367,38.79353),(-119.50413,38.90771),(-119.59459,38.96319),(-119.63466,39.00318),(-119.67473,39.06417),(-119.70119,39.14967),(-119.75488,39.19516),(-119.81497,39.25065),(-119.87506,39.31164),(-119.93515,39.37263),(-119.99524,39.40362),(-120.05533,39.44961),(-120.07179,39.45511),(-120.18805,39.4551),(-120.30431,39.45509),(-120.42057,39.44958),(-120.50503,39.44407),(-120.60189,39.43306),(-120.65558,39.43855),(-120.70927,39.45504),(-120.76296,39.44953),(-120.81665,39.45502),(-120.87034,39.45501),(-120.92403,39.45),(-120.97772,39.4505),(-121.07457,39.44499),(-121.11465,39.44498),(-121.12825,39.44498),(-121.14144,39.44498),(-121.14785,39.34789),(-121.16085,39.20981),(-121.16126,38.98515),(-121.14426,38.77599),(-121.06731,38.68441),(-120.99676,38.57573),(-120.99036,38.52024),(-120.99036,38.40056),(-120.98396,38.28088),(-120.9808,38.16119),(-120.82877,37.94281),(-120.76468,37.88732),(-120.70059,37.77314),(-120.52136,37.77863),(-120.48128,37.6595),(-120.32922,37.55032),(-120.32922,37.38669),(-120.28274,37.27801),(-120.13068,37.05963),(-119.98502,36.9789),(-119.83296,36.9844),(-119.52883,36.9844),(-119.23751,36.9844),(-119.14065,36.9844),(-119.03099,36.95939),(-118.98452,36.96489),(-118.92043,37.01488),(-118.87395,37.04539),(-118.8019,37.1101),(-118.72985,37.1751),(-118.66937,37.26018),(-118.59091,37.3013),(-118.5,37.34792),(-118.42154,37.41293),(-118.37507,37.47793),(-118.34179,37.55943),(-118.33539,37.70306),(-118.38828,37.80074),(-118.4348,37.95537),(-118.52526,38.0640),(-118.60973,38.17823),(-118.67382,38.28146),(-118.71389,38.36246),(-118.72029,38.48214),(-118.7331,38.59632),(-118.76997,38.70500),(-118.83406,38.78049),(-118.89815,38.84598),(-118.94463,38.91697),(-119.00232,39.0005),(-116.47215,36.44656)]
SD4_BOUNDARY = _SD4_RAW

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
def blend_environment(model_env: float, model_sd: float, polls: list) -> tuple:
    """
    Bayesian blend of structural model with poll average.
    Returns (blended_env, blended_sd).
    polls = list of {"dem_share": float, "n": int, "weight": float}
    """
    if not polls:
        return model_env, model_sd

    # Model precision
    model_prec = 1.0 / (model_sd ** 2)

    # Poll precision (combined)
    # Each poll contributes variance ~ p(1-p)/n, scaled by user weight
    poll_prec = 0.0
    poll_weighted_mean = 0.0
    for p in polls:
        dem = p["dem_share"]
        n   = max(p["n"], 1)
        w   = p.get("weight", 1.0)
        poll_var  = (dem * (1 - dem) / n) / max(w, 0.01)
        poll_prec += 1.0 / poll_var
        poll_weighted_mean += (1.0 / poll_var) * dem

    poll_mean = poll_weighted_mean / poll_prec if poll_prec > 0 else model_env
    total_prec = model_prec + poll_prec
    blended_mean = (model_prec * model_env + poll_prec * poll_mean) / total_prec
    blended_sd   = (1.0 / total_prec) ** 0.5

    return float(blended_mean), float(blended_sd)

# Compute blended state environment
model_env_val = st.session_state["model_forecast_env"] / 100
blended_env, blended_sd = blend_environment(
    model_env_val, STATE_ENV_SD, st.session_state["poll_entries"]
)

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
    with c3:
        ctx_trump_office = st.toggle("Trump in Office", value=True)
        ctx_trump_ballot = st.toggle("Trump on Ballot", value=False)
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_model = st.button("🔄 Run Forecast Model", type="primary", width="stretch")
        st.caption("Updates all parameters from historical CSVs using the context above.")

    if run_model:
        ctx_input = {
            "year": int(ctx_year), "general": bool(ctx_general),
            "presidential": bool(ctx_pres), "inflation": float(ctx_inflation),
            "approval": float(ctx_approval), "trump_in_office": bool(ctx_trump_office),
            "trump_on_ballot": bool(ctx_trump_ballot),
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
        "Add statewide CA polls (governor or senate) to blend with the structural model. "
        "The blend uses inverse-variance weighting: larger samples and higher weights pull "
        "the forecast further from the structural prior."
    )

    polls = st.session_state.get("poll_entries", [])

    with st.expander("➕ Add a poll", expanded=len(polls)==0):
        pc1,pc2,pc3,pc4,pc5 = st.columns([2,1,1,1,1])
        with pc1: poll_name  = st.text_input("Poll / source", placeholder="e.g. Emerson, PPIC")
        with pc2: poll_dem   = st.number_input("Dem share (%)", value=59.0, step=0.1, format="%.1f") / 100
        with pc3: poll_n     = st.number_input("Sample size", value=600, step=50)
        with pc4: poll_w     = st.number_input("Weight (1=full)", value=1.0, step=0.1, min_value=0.1, max_value=2.0, format="%.1f")
        with pc5:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add poll"):
                polls.append({"name": poll_name or f"Poll {len(polls)+1}", "dem_share": poll_dem, "n": int(poll_n), "weight": poll_w})
                st.session_state["poll_entries"] = polls
                st.rerun()

    if polls:
        blended_env2, blended_sd2 = blend_environment(model_env_val, STATE_ENV_SD, polls)
        poll_mean = sum(p["dem_share"]*p["weight"] for p in polls)/sum(p["weight"] for p in polls)

        st.markdown(f"""
        <div class="poll-blend">
          <b>Blended state environment: {blended_env2:.2%}</b> (SD={blended_sd2:.2%})<br>
          <span style="font-size:0.82rem;color:#555">
            Structural model: {model_env_val:.2%} · Poll average: {poll_mean:.2%} · Polls: {len(polls)}
          </span>
        </div>
        """, unsafe_allow_html=True)

        poll_rows=""
        for i,p in enumerate(polls):
            poll_rows += (f"<tr><td>{p['name']}</td><td>{fmt_pct(p['dem_share'])}</td>"
                          f"<td>{p['n']:,}</td><td>{p['weight']:.1f}</td>"
                          f"<td><form><button name='del_{i}' value='1' style='background:none;border:none;cursor:pointer;color:#b91c1c;font-size:0.8rem'>✕</button></form></td></tr>")
        st.markdown(f'<table class="styled-table"><thead><tr><th>Poll</th><th>Dem Share</th><th>N</th><th>Weight</th><th></th></tr></thead><tbody>{poll_rows}</tbody></table>', unsafe_allow_html=True)

        col_clear, _ = st.columns([1,3])
        with col_clear:
            if st.button("🗑 Clear all polls"):
                st.session_state["poll_entries"] = []; st.rerun()

        # Individual poll removal
        for i in range(len(polls)):
            if st.session_state.get(f"del_{i}"):
                polls.pop(i); st.session_state["poll_entries"]=polls; st.rerun()
    else:
        st.caption("No polls added yet. The simulation uses the structural model only.")

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

        # County fixed effects chart
        fe_names = {k:v for k,v in tcoef.items() if k not in ("intercept","presidential","general","prev_turnout")}
        if fe_names:
            fig_fe, ax_fe = plt.subplots(figsize=(10,3))
            fig_fe.patch.set_facecolor("#f7f7f5"); ax_fe.set_facecolor("#f7f7f5")
            names=list(fe_names.keys()); vals=[fe_names[n] for n in names]
            colors=["#1a6b3c" if v>=0 else "#b91c1c" for v in vals]
            ax_fe.bar(names,vals,color=colors,alpha=0.75)
            ax_fe.axhline(0,color="#ccc",linewidth=0.8)
            ax_fe.set_xticks(range(len(names))); ax_fe.set_xticklabels(names,rotation=40,ha="right",fontsize=7)
            ax_fe.set_ylabel("Fixed Effect (vs Alpine)",fontsize=8)
            ax_fe.set_title("County Fixed Effects — Turnout Model",fontsize=8,fontweight="600")
            ax_fe.spines[["top","right","left"]].set_visible(False); ax_fe.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig_fe,width="stretch"); plt.close()

    sd_c1, sd_c2 = st.columns(2)
    with sd_c1:
        st.markdown(f'<div class="stat-card"><div class="label">District Turnout SD</div><div class="value">{fmt_pct(sds2["district_turnout_sd"])}</div><div class="sub">Residual SD from turnout regression</div></div>', unsafe_allow_html=True)
    with sd_c2:
        st.markdown(f'<div class="stat-card"><div class="label">County Turnout SD</div><div class="value">{fmt_pct(sds2["county_turnout_sd"])}</div><div class="sub">Mean per-county residual SD</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Polling blend visualisation ───────────────────────────────────────────
    polls3 = st.session_state.get("poll_entries", [])
    if polls3:
        st.markdown('<div class="section-label">Polling Blend Illustration</div>', unsafe_allow_html=True)
        b_env, b_sd = blend_environment(model_env_val, STATE_ENV_SD, polls3)
        x_range = np.linspace(0.40, 0.75, 400)
        from scipy.stats import norm as _norm
        model_pdf  = _norm.pdf(x_range, model_env_val, STATE_ENV_SD)
        blend_pdf  = _norm.pdf(x_range, b_env, b_sd)
        fig_pb, ax_pb = plt.subplots(figsize=(9,3))
        fig_pb.patch.set_facecolor("#f7f7f5"); ax_pb.set_facecolor("#f7f7f5")
        ax_pb.plot(x_range*100, model_pdf,  color="#888", linewidth=1.5, linestyle="--", label=f"Structural model ({model_env_val:.1%})")
        ax_pb.plot(x_range*100, blend_pdf,  color="#1a6b3c", linewidth=2.0, label=f"Blended ({b_env:.1%}, SD={b_sd:.2%})")
        poll_avg = sum(p["dem_share"]*p["weight"] for p in polls3)/sum(p["weight"] for p in polls3)
        ax_pb.axvline(poll_avg*100, color="#d97706", linewidth=1.5, linestyle=":", label=f"Poll avg ({poll_avg:.1%})")
        ax_pb.set_xlabel("State Dem Share (%)", fontsize=8); ax_pb.set_ylabel("Density", fontsize=8)
        ax_pb.xaxis.set_major_formatter(mtick.PercentFormatter()); ax_pb.legend(fontsize=7, framealpha=0)
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
        v=st.number_input("District Turnout SD (%)",min_value=0.0,max_value=50.0,step=0.01,value=float(st.session_state["district_turnout_sd_pct"]),format="%.2f",key="in_dst_sd")
        st.session_state["district_turnout_sd_pct"]=v
    with sd_c2:
        v=st.number_input("County Turnout SD (%)",min_value=0.0,max_value=50.0,step=0.01,value=float(st.session_state["county_turnout_sd_pct"]),format="%.2f",key="in_cty_sd")
        st.session_state["county_turnout_sd_pct"]=v
    with sd_c3:
        v=st.number_input("State Env SD (%)",min_value=0.0,max_value=50.0,step=0.01,value=float(st.session_state["state_env_sd_pct"]),format="%.2f",key="in_env_sd")
        st.session_state["state_env_sd_pct"]=v
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
