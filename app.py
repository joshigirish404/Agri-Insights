"""
Streamlit Agri-Insights
--------------------------------------
Loads State/District/SubDistrict boundaries (GeoParquet), MODIS NDVI COGs (GeoTIFF)
and daily rainfall (actual + normal, Parquet) from a Hugging Face Storage Bucket
("girishhf07/agri_datasets").

Data layout expected in the HF Storage Bucket:
    boundaries/India_States.parquet
    boundaries/India_Districts.geoparquet
    boundaries/India_SubDistricts.geoparquet
    MOD13Q1_2026/MOD13Q1.061__250m_16_days_NDVI_doy2026<DDD>000000_aid0001.tif
    rainfal/district_rainfall_stats_2026.parquet      wide: STATE, STATE_LGD, DISTRICT, DIST_LGD, 01-01-2026, 02-01-2026, ...
    rainfal/subdistrict_rainfall_stats_2026.parquet   wide: STATE, STATE_LGD, DISTRICT, DIST_LGD, SUB_DIST, ..., 01-01-2026, ...
    rainfal/district_rainfall_normal.parquet          wide: STATE, STATE_LGD, DISTRICT, DIST_LGD, 01-01, 01-02, ...   (MM-DD)
    rainfal/subdistrict_rainfall_normal.parquet       wide: ... SUB_DIST ..., 01-01, 01-02, ...                       (MM-DD)

Layout:
    Main window (top)  : title + short description
                         State / District / SubDistrict MULTI-select filters + Reset button
    Main window        : NDVI map (collapsible)
                         Rainfall section (collapsible): deviation-from-normal map,
                         "Plot Graph" button below it, then the graph (actual solid, normal dotted)
    Sidebar            : collapsible "NDVI" and "Rainfall" sections
    Page end           : data-source citations

Selection logic (the DEEPEST level that has a selection decides what is plotted):
    SubDistrict(s) selected -> NDVI clipped to / rainfall for exactly those subdistricts
    District(s) selected    -> exactly those districts
    State(s) selected       -> exactly those states
                               (state rainfall = area-weighted mean of its districts' daily means)
    nothing selected        -> All India view (map only; pick a State for NDVI / rainfall)

Rainfall design notes:
    * Only DAILY rainfall is stored. Weekly and monthly values are cumulative sums of daily data.
    * Weeks are fixed 7-day blocks counted from 01 Jan (Week 1 = 01 Jan - 07 Jan, ...).
    * Normal rainfall is compared like-for-like: only normal days up to the last day for which
      actual data exists are summed (so a part-complete month is not compared with a full-month normal).
    * Deviation (%) = (Actual - Normal) / Normal * 100 over the selected range, classified with the
      IMD categories (see DEV_* constants).
"""

import base64
import calendar
import html as _html
import io
import os
import re
import tempfile
import time
from datetime import date, timedelta

import folium
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import shapely
import rasterio
import streamlit as st
from huggingface_hub import download_bucket_files, list_bucket_tree
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping
from streamlit_folium import st_folium

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Agri-Insights", layout="wide", initial_sidebar_state="expanded")

HF_BUCKET_ID = "girishhf07/agri_datasets"

STATE_PARQUET_PATH = "boundaries/India_States.parquet"
DISTRICT_PARQUET_PATH = "boundaries/India_Districts.geoparquet"
SUBDISTRICT_PARQUET_PATH = "boundaries/India_SubDistricts.geoparquet"
NDVI_FOLDER = "MOD13Q1_2026"

RAIN_FOLDER = "rainfall"
DISTRICT_RAIN_PATH = f"{RAIN_FOLDER}/district_rainfall_stats_2026.parquet"
SUBDISTRICT_RAIN_PATH = f"{RAIN_FOLDER}/subdistrict_rainfall_stats_2026.parquet"
DISTRICT_NORMAL_PATH = f"{RAIN_FOLDER}/district_rainfall_normal.parquet"
SUBDISTRICT_NORMAL_PATH = f"{RAIN_FOLDER}/subdistrict_rainfall_normal.parquet"
RAIN_CACHE_MAX_AGE_S = 6 * 3600  # actual rainfall grows daily -> re-download if local copy is older than this
NDVI_LISTING_TTL_S = 3600        # how long the list of available NDVI files is cached

CACHE_DIR = os.path.join(tempfile.gettempdir(), "modis_ndvi_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = None

RASTER_NODATA_FILL = -3000  # MODIS NDVI fill value
NDVI_SCALE = 0.0001
NDVI_VALID_MIN, NDVI_VALID_MAX = -0.2, 1.0  # after scaling

years_list = [2026]  # fallback only; real years come from the files found in the bucket

fn_dict = {
    "Jan 01 - Jan 16": 1, "Jan 17 - Feb 01": 2, "Feb 02 - Feb 17": 3, "Feb 18 - Mar 05": 4,
    "Mar 06 - Mar 21": 5, "Mar 22 - Apr 06": 6, "Apr 07 - Apr 22": 7, "Apr 23 - May 08": 8,
    "May 09 - May 24": 9, "May 25 - Jun 09": 10, "Jun 10 - Jun 25": 11, "Jun 26 - Jul 11": 12,
    "Jul 12 - Jul 27": 13, "Jul 28 - Aug 12": 14, "Aug 13 - Aug 28": 15, "Aug 29 - Sep 13": 16,
    "Sep 14 - Sep 29": 17, "Sep 30 - Oct 15": 18, "Oct 16 - Oct 31": 19, "Nov 01 - Nov 16": 20,
    "Nov 17 - Dec 02": 21, "Dec 03 - Dec 20": 22, "Dec 21 - Dec 31": 23,
}

julian_dates_dict = {
    1: "001", 2: "017", 3: "033", 4: "049", 5: "065", 6: "081", 7: "097", 8: "113",
    9: "129", 10: "145", 11: "161", 12: "177", 13: "193", 14: "209", 15: "225", 16: "241",
    17: "257", 18: "273", 19: "289", 20: "305", 21: "321", 22: "337", 23: "353",
}

INDIA_CENTER = [22.5937, 80.9629]
INDIA_ZOOM = 4

# Free, key-less satellite basemap (Esri World Imagery). Cite the source in your app/report.
ESRI_IMAGERY_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_IMAGERY_ATTR = (
    "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, "
    "and the GIS User Community"
)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "VSI_CACHE": "TRUE",
}

FLT_KEYS = ("flt_states", "flt_districts", "flt_subs")

# Selected units are drawn with the FULL-resolution boundary (coordinates snapped to this grid,
# 1e-5 deg ~ 1.1 m, to keep the map payload small). Context layers stay simplified.
# If more than EXACT_MAX_FEATURES units are selected, the simplified boundary is used instead.
EXACT_PRECISION = 1e-5
EXACT_MAX_FEATURES = 200

# ---- Rainfall deviation categories (IMD style) ----
DEV_ORDER = ["No Data", "No Rain", "Unlisted", "Large Deficient", "Deficient", "Normal", "Excess", "Large Excess"]
DEV_COLOR = {
    "No Data": "#d9d9d9",
    "No Rain": "#ffffff",
    "Unlisted": "#f5ebc3",
    "Large Deficient": "#ffff00",
    "Deficient": "#ff0000",
    "Normal": "#00ff00",
    "Excess": "#5bc8f0",
    "Large Excess": "#3b8fd9",
}
DEV_LEGEND_TEXT = {
    "No Data": "No Data",
    "No Rain": "No Rain",
    "Unlisted": "Unlisted",
    "Large Deficient": "Large Deficient [-99% to -60%]",
    "Deficient": "Deficient [-59% to -20%]",
    "Normal": "Normal [-19% to 19%]",
    "Excess": "Excess [20% to 59%]",
    "Large Excess": "Large Excess [60% or more]",
}

PAGE_CSS = """
<style>
/* ---- tighter top of the page ---- */
.block-container, div[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

/* ---- bold filter labels ---- */
section[data-testid="stSidebar"] label p,
div[data-testid="stMultiSelect"] label p {
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}
</style>
"""

APP_DESCRIPTION = (
    "An interactive dashboard for assessing agricultural conditions across India using "
    "NDVI (Normalized Difference Vegetation Index) and rainfall data. Monitor vegetation health, crop stages, "
    "analyze rainfall patterns, and compare actual rainfall with normal rainfall "
    "to identify potential agricultural stress and regional variations."
)


def citation_html() -> str:
    accessed = date.today().strftime("%d %B %Y")
    return f"""
<hr style="margin:1.5rem 0 0.4rem 0;">
<div style="font-size:0.72rem; line-height:1.4; color:#8a8a8a;">
  <div style="font-size:0.8rem; font-weight:700; margin-bottom:3px;">Data sources &amp; citations</div>
  <div style="margin-bottom:3px;"><b>NDVI:</b>
    Didan, K. (2021). <i>MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061</i> (MOD13Q1)
    [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center (LP DAAC).
    <a href="https://doi.org/10.5067/MODIS/MOD13Q1.061" target="_blank">https://doi.org/10.5067/MODIS/MOD13Q1.061</a>
    (accessed {accessed}).
  </div>
  <div style="margin-bottom:3px;"><b>Rainfall:</b>
    India Meteorological Department (IMD), Pune. Daily gridded rainfall data, 0.25&deg; &times; 0.25&deg;.
    Dataset described in: Pai, D. S., Sridhar, L., Rajeevan, M., Sreejith, O. P., Satbhai, N. S., &amp;
    Mukhopadhyay, B. (2014). Development of a new high spatial resolution (0.25&deg; &times; 0.25&deg;) long period
    (1901&ndash;2010) daily gridded rainfall data set over India and its comparison with existing data sets over
    the region. <i>MAUSAM, 65</i>(1), 1&ndash;18.
    <a href="https://doi.org/10.54302/mausam.v65i1.851" target="_blank">https://doi.org/10.54302/mausam.v65i1.851</a>.
    District/SubDistrict averages and normals shown here are derived from this dataset.
  </div>
  <div style="margin-bottom:3px;"><b>Basemap:</b> Esri World Imagery &mdash; Esri, Maxar, Earthstar Geographics, and the GIS User Community.</div>
  <div><b>Administrative Boundaries:</b> Survey of India, Government of India. Administrative boundaries are simplified for faster loading and visualisation purposes only
    (areas selected by the user are drawn with the full-resolution boundary).</div>
</div>
"""


# --------------------------------------------------------------------------------------
# DOWNLOAD HELPERS
# --------------------------------------------------------------------------------------
def _download_bucket_file(bucket_path: str) -> str:
    """Download one file from the HF Storage Bucket to a local cache path (skips
    re-download if already cached on disk from an earlier run in this container)."""
    local_path = os.path.join(CACHE_DIR, bucket_path.replace("/", "__"))
    if not os.path.exists(local_path):
        download_bucket_files(
            HF_BUCKET_ID,
            files=[(bucket_path, local_path)],
            token=HF_TOKEN,
        )
    return local_path


def _download_bucket_file_fresh(bucket_path: str, max_age_s: int = RAIN_CACHE_MAX_AGE_S) -> str:
    """Like _download_bucket_file, but re-downloads if the cached copy is older than
    max_age_s. Used for the actual-rainfall files, which are appended to as new days arrive."""
    local_path = os.path.join(CACHE_DIR, bucket_path.replace("/", "__"))
    stale = (not os.path.exists(local_path)) or (time.time() - os.path.getmtime(local_path) > max_age_s)
    if stale:
        if os.path.exists(local_path):
            os.remove(local_path)
        download_bucket_files(
            HF_BUCKET_ID,
            files=[(bucket_path, local_path)],
            token=HF_TOKEN,
        )
    return local_path


# --------------------------------------------------------------------------------------
# DATA LOADING (boundaries) — cached so they're only fetched once per session
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading state boundaries...")
def load_states() -> gpd.GeoDataFrame:
    path = _download_bucket_file(STATE_PARQUET_PATH)
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


@st.cache_resource(show_spinner="Loading district boundaries...")
def load_districts() -> gpd.GeoDataFrame:
    path = _download_bucket_file(DISTRICT_PARQUET_PATH)
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


@st.cache_resource(show_spinner="Loading subdistrict boundaries...")
def load_subdistricts() -> gpd.GeoDataFrame:
    path = _download_bucket_file(SUBDISTRICT_PARQUET_PATH)
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


@st.cache_resource(show_spinner=False)
def state_list() -> list:
    """Excludes any 'Disputed...' entries in STATE."""
    states = load_states()["STATE"].unique().tolist()
    states = [s for s in states if "DISPUTED" not in str(s)]
    return sorted(states)


@st.cache_resource(show_spinner=False)
def district_lookup() -> "dict[str, list]":
    df = load_districts()[["STATE", "DISTRICT"]]
    return {
        state: sorted(group["DISTRICT"].dropna().unique().tolist())
        for state, group in df.groupby("STATE")
    }


@st.cache_resource(show_spinner=False)
def india_states_geojson_simplified() -> dict:
    gdf = load_states()[["STATE", "geometry"]].copy()
    gdf["geometry"] = gdf.geometry.simplify(0.01, preserve_topology=True)
    return gdf.__geo_interface__


@st.cache_resource(show_spinner=False)
def districts_geojson_simplified_by_state() -> "dict[str, dict]":
    gdf = load_districts()[["STATE", "DISTRICT", "geometry"]].copy()
    gdf["geometry"] = gdf.geometry.simplify(0.005, preserve_topology=True)
    return {
        state: group.__geo_interface__
        for state, group in gdf.groupby("STATE")
    }


@st.cache_resource(show_spinner=False)
def state_bounds_lookup() -> "dict[str, tuple]":
    gdf = load_states()
    return {state: group.total_bounds for state, group in gdf.groupby("STATE")}


@st.cache_resource(show_spinner=False)
def district_bounds_lookup() -> "dict[tuple, tuple]":
    gdf = load_districts()
    return {
        (state, district): group.total_bounds
        for (state, district), group in gdf.groupby(["STATE", "DISTRICT"])
    }


@st.cache_resource(show_spinner=False)
def subdistrict_lookup() -> "dict[tuple, list]":
    df = load_subdistricts()[["STATE", "DISTRICT", "SUB_DIST"]]
    return {
        key: sorted(group["SUB_DIST"].dropna().unique().tolist())
        for key, group in df.groupby(["STATE", "DISTRICT"])
    }


@st.cache_resource(show_spinner=False)
def subdistricts_geojson_simplified_by_district() -> "dict[tuple, dict]":
    gdf = load_subdistricts()[["STATE", "DISTRICT", "SUB_DIST", "geometry"]].copy()
    gdf["geometry"] = gdf.geometry.simplify(0.003, preserve_topology=True)
    return {
        key: group.__geo_interface__
        for key, group in gdf.groupby(["STATE", "DISTRICT"])
    }


@st.cache_resource(show_spinner=False)
def subdistrict_bounds_lookup() -> "dict[tuple, tuple]":
    gdf = load_subdistricts()
    return {
        (state, district, subdistrict): group.total_bounds
        for (state, district, subdistrict), group in gdf.groupby(["STATE", "DISTRICT", "SUB_DIST"])
    }


# --------------------------------------------------------------------------------------
# MAP LABEL POINTS (always-on labels) — one point at the centre of each feature
# --------------------------------------------------------------------------------------
def _label_frame(gdf: gpd.GeoDataFrame, name_cols: list) -> pd.DataFrame:
    """One label point per unit. Uses the true centroid (computed in an equal-area CRS so it is
    not distorted); if the centroid falls outside the polygon (crescent shapes, islands,
    multi-part units) it falls back to a point guaranteed to be inside the polygon."""
    g = gdf[name_cols + ["geometry"]].dropna(subset=name_cols).reset_index(drop=True)
    if g.duplicated(name_cols).any():          # a unit stored as several rows -> merge first
        g = g.dissolve(by=name_cols, as_index=False).reset_index(drop=True)

    proj = g.to_crs("EPSG:6933")
    geoms = proj.geometry
    cen = geoms.centroid
    rep = geoms.representative_point()
    inside = cen.within(geoms).values
    x = np.where(inside, cen.x.values, rep.x.values)
    y = np.where(inside, cen.y.values, rep.y.values)
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=proj.crs).to_crs("EPSG:4326")

    df = g[name_cols].copy()
    df["lat"] = pts.y.values
    df["lon"] = pts.x.values
    return df.drop_duplicates(name_cols)


@st.cache_resource(show_spinner=False)
def state_labels() -> list:
    """[(STATE, lat, lon), ...] for every (non-disputed) state."""
    df = _label_frame(load_states(), ["STATE"])
    df = df[~df["STATE"].astype(str).str.contains("DISPUTED")]
    return list(zip(df["STATE"], df["lat"], df["lon"]))


@st.cache_resource(show_spinner=False)
def district_labels_by_state() -> "dict[str, list]":
    df = _label_frame(load_districts(), ["STATE", "DISTRICT"])
    return {
        state: list(zip(g["DISTRICT"], g["lat"], g["lon"]))
        for state, g in df.groupby("STATE")
    }


@st.cache_resource(show_spinner=False)
def subdistrict_labels_by_district() -> "dict[tuple, list]":
    df = _label_frame(load_subdistricts(), ["STATE", "DISTRICT", "SUB_DIST"])
    return {
        key: list(zip(g["SUB_DIST"], g["lat"], g["lon"]))
        for key, g in df.groupby(["STATE", "DISTRICT"])
    }


# --------------------------------------------------------------------------------------
# NDVI AVAILABILITY (which fortnights actually exist in the bucket)
# --------------------------------------------------------------------------------------
_NDVI_FILE_RE = re.compile(r"250m_16_days_NDVI_doy(\d{4})(\d{3})")


@st.cache_data(show_spinner=False, ttl=NDVI_LISTING_TTL_S)
def list_available_ndvi() -> "dict[int, list[int]]":
    """{year: [sorted day-of-year starts]} for NDVI GeoTIFFs present in NDVI_FOLDER.
    Raises on listing failure (so a failure is never cached)."""
    out: dict = {}
    for item in list_bucket_tree(HF_BUCKET_ID, prefix=NDVI_FOLDER, recursive=True, token=HF_TOKEN):
        path = str(getattr(item, "path", "") or "")
        if not path.lower().endswith(".tif"):
            continue
        m = _NDVI_FILE_RE.search(os.path.basename(path))
        if m:
            out.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    return {y: sorted(v) for y, v in out.items()}


def available_fortnights(year: int, avail: "dict[int, list[int]]") -> list:
    """Fortnight labels (in calendar order) whose file exists for `year`."""
    doys = set(avail.get(year, []))
    return [lbl for lbl, n in fn_dict.items() if int(julian_dates_dict[n]) in doys]


# --------------------------------------------------------------------------------------
# RASTER FILENAME RESOLUTION  (NDVI workflow — unchanged)
# --------------------------------------------------------------------------------------
def build_raster_filename(year: int, fortnight_label: str) -> str:
    julian_day = julian_dates_dict[fn_dict[fortnight_label]]
    doy = f"{year}{julian_day}000000"
    return f"MOD13Q1.061__250m_16_days_NDVI_doy{doy}_aid0001.tif"


def build_raster_bucket_path(filename: str) -> str:
    return f"{NDVI_FOLDER}/{filename}"


@st.cache_resource(show_spinner="Fetching NDVI raster...")
def get_local_raster_path(bucket_path: str) -> str:
    return _download_bucket_file(bucket_path)


# --------------------------------------------------------------------------------------
# WINDOWED READ + CLIP + REPROJECT  (NDVI workflow — unchanged)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_and_clip_ndvi(local_raster_path: str, clip_wkt: str, clip_crs: str):
    geom = gpd.GeoSeries.from_wkt([clip_wkt], crs=clip_crs)

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(local_raster_path) as src:
            geom_native = geom.to_crs(src.crs)
            clipped, clipped_transform = rio_mask(
                src, geom_native.geometry, crop=True, filled=True, nodata=RASTER_NODATA_FILL
            )
            src_crs = src.crs
            src_nodata = RASTER_NODATA_FILL

    band = clipped[0].astype("float64")
    band[band == src_nodata] = np.nan
    ndvi = band * NDVI_SCALE

    height, width = ndvi.shape
    dst_crs = "EPSG:4326"
    left, bottom, right, top = rasterio.transform.array_bounds(height, width, clipped_transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, left=left, bottom=bottom, right=right, top=top
    )
    dst_array = np.full((dst_height, dst_width), np.nan, dtype="float64")
    reproject(
        source=ndvi,
        destination=dst_array,
        src_transform=clipped_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    d_left, d_bottom, d_right, d_top = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
    bounds_4326 = [[d_bottom, d_left], [d_top, d_right]]

    return dst_array, bounds_4326


def array_to_png_base64(arr: np.ndarray, vmin=NDVI_VALID_MIN, vmax=NDVI_VALID_MAX) -> str:
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    cmap = matplotlib.colormaps["RdYlGn"]
    rgba = cmap(norm)
    rgba[np.isnan(arr)] = (0, 0, 0, 0)
    rgba_uint8 = (rgba * 255).astype("uint8")

    from PIL import Image
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def clipped_array_to_geotiff_bytes(arr: np.ndarray, bounds_4326) -> bytes:
    (south, west), (north, east) = bounds_4326
    height, width = arr.shape
    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)
    out_arr = np.where(np.isnan(arr), -9999.0, arr).astype("float32")

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=height, width=width, count=1, dtype="float32",
            crs="EPSG:4326", transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(out_arr, 1)
        return memfile.read()


def ndvi_legend_html(vmin=NDVI_VALID_MIN, vmax=NDVI_VALID_MAX) -> str:
    cmap = matplotlib.colormaps["RdYlGn"]
    n_stops = 10
    stops = []
    for i in range(n_stops + 1):
        frac = i / n_stops
        r, g, b, _ = cmap(frac)
        stops.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)}) {frac*100:.0f}%")
    gradient_css = ", ".join(stops)

    tick_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    ticks_html = "".join(f"<span>{v:.2f}</span>" for v in tick_values)

    return f"""
    <div style="
        position: fixed; bottom: 30px; right: 20px; z-index: 9999;
        background: rgba(255,255,255,0.92); padding: 8px 10px 6px 10px;
        border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.35);
        font-family: sans-serif; font-size: 12px; width: 160px;">
        <div style="font-weight:700; margin-bottom:4px;">NDVI</div>
        <div style="width:100%; height:12px; border-radius:2px;
                    background: linear-gradient(to right, {gradient_css});
                    border: 1px solid #999;"></div>
        <div style="display:flex; justify-content:space-between; font-size:10px; margin-top:3px;">
            {ticks_html}
        </div>
    </div>
    """


def rain_legend_html() -> str:
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="display:inline-block;width:14px;height:14px;background:{DEV_COLOR[c]};'
        f'border:1px solid #888;margin-right:6px;"></span>{_html.escape(DEV_LEGEND_TEXT[c])}</div>'
        for c in DEV_ORDER
    )
    return f"""
    <div style="
        position: fixed; bottom: 30px; left: 20px; z-index: 9999;
        background: rgba(255,255,255,0.92); padding: 8px 10px;
        border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.35);
        font-family: sans-serif; font-size: 12px; color: #222;">
        <div style="font-weight:700; margin-bottom:4px;">Rainfall deviation from Normal</div>
        {rows}
    </div>
    """


# --------------------------------------------------------------------------------------
# SELECTION HELPERS
# --------------------------------------------------------------------------------------
def get_selection() -> dict:
    """Current filter state. 'level' = deepest level that has a selection."""
    states = list(st.session_state.get("flt_states", []))
    districts = list(st.session_state.get("flt_districts", []))   # [(state, district)]
    subs = list(st.session_state.get("flt_subs", []))             # [(state, district, subdistrict)]
    if subs:
        level = "subdistrict"
    elif districts:
        level = "district"
    elif states:
        level = "state"
    else:
        level = "india"
    return {"level": level, "states": states, "districts": districts, "subs": subs}


def selection_names(sel: dict) -> list:
    if sel["level"] == "state":
        return list(sel["states"])
    if sel["level"] == "district":
        return [d for _, d in sel["districts"]]
    if sel["level"] == "subdistrict":
        return [s for _, _, s in sel["subs"]]
    return []


def summarize_names(names: list, max_n: int = 3) -> str:
    if len(names) <= max_n:
        return ", ".join(names)
    return f"{', '.join(names[:max_n])} +{len(names) - max_n} more"


def selection_geometry(sel: dict):
    """(union geometry, crs string) of all selected units at the deepest selected level."""
    level = sel["level"]
    if level == "state":
        gdf = load_states()
        mask = gdf["STATE"].isin(sel["states"]).values
    elif level == "district":
        gdf = load_districts()
        mask = pd.MultiIndex.from_frame(gdf[["STATE", "DISTRICT"]]).isin(sel["districts"])
    elif level == "subdistrict":
        gdf = load_subdistricts()
        mask = pd.MultiIndex.from_frame(gdf[["STATE", "DISTRICT", "SUB_DIST"]]).isin(sel["subs"])
    else:
        return None, None
    geoms = gdf.loc[mask].geometry
    if geoms.empty:
        return None, None
    union = geoms.union_all() if hasattr(geoms, "union_all") else geoms.unary_union
    return union, str(gdf.crs)


def selection_bounds(sel: dict):
    """[[south, west], [north, east]] covering all selected units, or None (India view)."""
    level = sel["level"]
    if level == "state":
        bs = [state_bounds_lookup().get(s) for s in sel["states"]]
    elif level == "district":
        bs = [district_bounds_lookup().get(k) for k in sel["districts"]]
    elif level == "subdistrict":
        bs = [subdistrict_bounds_lookup().get(k) for k in sel["subs"]]
    else:
        return None
    bs = [b for b in bs if b is not None]
    if not bs:
        return None
    arr = np.array(bs)
    return [[float(arr[:, 1].min()), float(arr[:, 0].min())],
            [float(arr[:, 3].max()), float(arr[:, 2].max())]]


# --------------------------------------------------------------------------------------
# RAINFALL: loading (wide parquet -> tidy daily), period helpers, aggregation, deviation, plotting
# --------------------------------------------------------------------------------------
_ACTUAL_DATE_FORMATS = ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M")
_MD_RE = re.compile(r"^\s*(\d{1,2})[-/](\d{1,2})\s*$")


def _actual_col_date(col):
    """Column header -> Timestamp for dd-mm-yyyy style headers; NaT for non-date columns."""
    if isinstance(col, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(col).normalize()
    s = str(col).strip()
    for fmt in _ACTUAL_DATE_FORMATS:
        t = pd.to_datetime(s, format=fmt, errors="coerce")
        if pd.notna(t):
            return t.normalize()
    return pd.NaT


def _normal_col_date(col, year: int):
    """'MM-DD' header -> Timestamp in `year` (NaT for non-date columns and for 29 Feb in non-leap years)."""
    m = _MD_RE.match(str(col))
    if not m:
        return pd.NaT
    try:
        return pd.Timestamp(year, int(m.group(1)), int(m.group(2)))
    except ValueError:
        return pd.NaT


def _wide_to_long(df: pd.DataFrame, level: str, col_dates: dict) -> pd.DataFrame:
    """Melt a wide table (one column per date) into STATE, DISTRICT, [SUB_DIST], date, rain_mm."""
    keep = ["STATE", "DISTRICT"] + (["SUB_DIST"] if level == "subdistrict" else [])
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing}; found {list(df.columns)[:12]} ...")
    date_cols = [c for c, d in col_dates.items() if pd.notna(d)]
    if not date_cols:
        raise ValueError("No date columns recognised in the parquet file.")

    work = df[keep + date_cols].copy()
    for c in keep:
        work[c] = work[c].astype(str).str.strip()

    long = work.melt(id_vars=keep, value_vars=date_cols, var_name="_col", value_name="rain_mm")
    long["date"] = long["_col"].map({c: col_dates[c] for c in date_cols})
    long["rain_mm"] = pd.to_numeric(long["rain_mm"], errors="coerce")
    long = long.drop(columns="_col").dropna(subset=["date"])
    long["date"] = pd.to_datetime(long["date"])
    return add_period_columns(long.reset_index(drop=True))


@st.cache_resource(show_spinner=False, ttl=RAIN_CACHE_MAX_AGE_S)
def load_rainfall(level: str) -> pd.DataFrame:
    """ACTUAL daily rainfall. level = 'district' or 'subdistrict'.
    Reads the wide parquet (columns like 01-01-2026) and returns a tidy DataFrame:
    STATE, DISTRICT, [SUB_DIST], date, rain_mm, doy, week, month."""
    bucket_path = DISTRICT_RAIN_PATH if level == "district" else SUBDISTRICT_RAIN_PATH
    path = _download_bucket_file_fresh(bucket_path)
    df = pd.read_parquet(path)
    df.columns = [str(c).strip() for c in df.columns]
    col_dates = {c: _actual_col_date(c) for c in df.columns}
    out = _wide_to_long(df, level, col_dates)
    if out.empty:
        raise ValueError("No valid rainfall rows found. Expected date headers like dd-mm-yyyy.")
    return out


@st.cache_resource(show_spinner=False)
def load_rainfall_normal(level: str, year: int) -> pd.DataFrame:
    """NORMAL daily rainfall. Wide parquet with MM-DD headers is mapped onto the dates of `year`
    (so it lines up with the actual data on day-of-year / week / month keys)."""
    bucket_path = DISTRICT_NORMAL_PATH if level == "district" else SUBDISTRICT_NORMAL_PATH
    path = _download_bucket_file(bucket_path)
    df = pd.read_parquet(path)
    df.columns = [str(c).strip() for c in df.columns]
    col_dates = {c: _normal_col_date(c, year) for c in df.columns}
    out = _wide_to_long(df, level, col_dates)
    if out.empty:
        raise ValueError("No valid normal-rainfall rows found. Expected MM-DD headers.")
    return out


def add_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Integer period keys. Keying on day-of-year / week / month number (not on dates)
    lets the Actual and Normal series be aggregated identically and joined on the same keys."""
    if "doy" in df.columns:  # already computed once at load time
        return df
    out = df.copy()
    out["doy"] = out["date"].dt.dayofyear
    # Week 1 = day 1-7, Week 2 = day 8-14, ... ; only 52 weeks: the leftover day(s)
    # after day 364 (31 Dec, plus 31 Dec/leap day in leap years) belong to Week 52.
    out["week"] = np.minimum((out["doy"] - 1) // 7 + 1, 52)
    out["month"] = out["date"].dt.month
    return out


@st.cache_resource(show_spinner=False)
def district_area_lookup() -> "dict[tuple, float]":
    """{(STATE.upper(), DISTRICT.upper()): area_km2} from the district boundaries (equal-area
    projection). Used to area-weight district rainfall when rolling it up to state level."""
    gdf = load_districts()[["STATE", "DISTRICT", "geometry"]].copy()
    gdf["area"] = gdf.to_crs("EPSG:6933").geometry.area / 1e6
    gdf["k_s"] = gdf["STATE"].astype(str).str.strip().str.upper()
    gdf["k_d"] = gdf["DISTRICT"].astype(str).str.strip().str.upper()
    tot = gdf.groupby(["k_s", "k_d"])["area"].sum()
    return {k: float(v) for k, v in tot.items()}


def _rollup_to_state(dist_df: pd.DataFrame) -> pd.DataFrame:
    """Daily STATE rainfall = area-weighted mean of the districts' daily mean rainfall.
    If any district of a state cannot be matched to a boundary polygon by name, that state
    falls back to a simple (equal-weight) mean of its districts.
    Columns: STATE, date, rain_mm, doy, week, month."""
    df = dist_df[["STATE", "DISTRICT", "date", "rain_mm"]].copy()
    areas = district_area_lookup()

    pairs = df[["STATE", "DISTRICT"]].drop_duplicates().copy()
    pairs["w"] = [
        areas.get((str(s).strip().upper(), str(d).strip().upper()), np.nan)
        for s, d in zip(pairs["STATE"], pairs["DISTRICT"])
    ]
    df = df.merge(pairs, on=["STATE", "DISTRICT"], how="left")

    has_na = df["w"].isna().groupby(df["STATE"]).transform("any")
    df.loc[has_na, "w"] = 1.0

    valid = df["rain_mm"].notna()
    df["rw"] = df["rain_mm"].fillna(0.0) * df["w"]
    df["wv"] = df["w"].where(valid, 0.0)
    g = df.groupby(["STATE", "date"], as_index=False)[["rw", "wv"]].sum()
    g["rain_mm"] = g["rw"] / g["wv"].where(g["wv"] > 0)
    out = g[["STATE", "date", "rain_mm"]].reset_index(drop=True)
    return add_period_columns(out)


@st.cache_resource(show_spinner=False, ttl=RAIN_CACHE_MAX_AGE_S)
def load_state_rainfall() -> pd.DataFrame:
    return _rollup_to_state(load_rainfall("district"))


@st.cache_resource(show_spinner=False)
def load_state_normal(year: int) -> pd.DataFrame:
    return _rollup_to_state(load_rainfall_normal("district", year))


def week_bounds(year: int, week_no: int):
    start = date(year, 1, 1) + timedelta(days=7 * (week_no - 1))
    end = date(year, 12, 31) if week_no == 52 else start + timedelta(days=6)
    return start, end


def week_label(year: int, week_no: int) -> str:
    start, end = week_bounds(year, week_no)
    if start == end:
        return f"Week {week_no} ({start:%d %b})"
    return f"Week {week_no} ({start:%d %b} - {end:%d %b})"


def month_label(month_no: int) -> str:
    return calendar.month_name[month_no]


def period_label(mode: str, key: int, year: int):
    """x-axis value for a period key."""
    if mode == "daily":
        return pd.Timestamp(date(year, 1, 1) + timedelta(days=int(key) - 1))
    if mode == "weekly":
        return week_label(year, int(key))
    return month_label(int(key))


def month_bounds(year: int, month_no: int):
    return date(year, month_no, 1), date(year, month_no, calendar.monthrange(year, month_no)[1])


def aggregate_rainfall(df: pd.DataFrame, entity_col: str, mode: str, start, end) -> pd.DataFrame:
    """Cumulative (sum) rainfall per entity per period within the selected range.

    mode='daily'   : start/end are dates (inclusive); one row per entity per day
    mode='weekly'  : start/end are week numbers (inclusive); sum of daily values per week
    mode='monthly' : start/end are month numbers (inclusive); sum of daily values per month

    Returns columns: entity, period (int key), rain_mm.
    """
    d = add_period_columns(df)
    if mode == "daily":
        mask = (d["date"] >= pd.Timestamp(start)) & (d["date"] <= pd.Timestamp(end))
        period_col = "doy"
    elif mode == "weekly":
        mask = (d["week"] >= start) & (d["week"] <= end)
        period_col = "week"
    else:
        mask = (d["month"] >= start) & (d["month"] <= end)
        period_col = "month"

    out = (
        d.loc[mask]
        .groupby([entity_col, period_col], as_index=False)["rain_mm"]
        .sum(min_count=1)
        .rename(columns={entity_col: "entity", period_col: "period"})
        .sort_values(["entity", "period"])
    )
    return out


def classify_deviation(actual, normal):
    """(category, deviation %) for one area. Deviation = (actual - normal) / normal * 100.
    No Data  : actual or normal missing
    No Rain  : no rainfall at all in the period (deviation -100 %)
    Unlisted : rain fell but the normal is zero, so a % deviation cannot be computed
    Otherwise IMD classes: Large Deficient <= -60, Deficient -59..-20, Normal -19..19,
    Excess 20..59, Large Excess >= 60."""
    if pd.isna(actual) or pd.isna(normal):
        return "No Data", np.nan
    if actual <= 0:
        return "No Rain", -100.0
    if normal <= 0:
        return "Unlisted", np.nan
    dev = (actual - normal) / normal * 100.0
    if dev <= -60:
        cat = "Large Deficient"
    elif dev <= -20:
        cat = "Deficient"
    elif dev < 20:
        cat = "Normal"
    elif dev < 60:
        cat = "Excess"
    else:
        cat = "Large Excess"
    return cat, dev


def _fmt_mm(v) -> str:
    return "—" if pd.isna(v) else f"{v:.1f} mm"


def _fmt_dev(v) -> str:
    return "—" if pd.isna(v) else f"{v:+.1f} %"


def _simplified_features(level: str, keys) -> list:
    """[(geojson feature, id-tuple)] from the simplified boundaries for the given unit keys.
    Keys are (STATE,), (STATE, DISTRICT) or (STATE, DISTRICT, SUB_DIST)."""
    keys = set(keys)
    out = []
    if level == "state":
        for f in india_states_geojson_simplified().get("features", []):
            k = (f["properties"].get("STATE"),)
            if k in keys:
                out.append((f, k))
    elif level == "district":
        by_state = districts_geojson_simplified_by_state()
        for s in {k[0] for k in keys}:
            for f in by_state.get(s, {}).get("features", []):
                k = (s, f["properties"].get("DISTRICT"))
                if k in keys:
                    out.append((f, k))
    else:
        by_dist = subdistricts_geojson_simplified_by_district()
        for sd in {(k[0], k[1]) for k in keys}:
            for f in by_dist.get(sd, {}).get("features", []):
                k = (sd[0], sd[1], f["properties"].get("SUB_DIST"))
                if k in keys:
                    out.append((f, k))
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def _exact_features_cached(level: str, keys: tuple) -> list:
    """[(geojson feature, id-tuple)] using the ORIGINAL (unsimplified) boundaries of the given units,
    coordinates snapped to EXACT_PRECISION. Cached per selection."""
    if level == "state":
        gdf, cols = load_states(), ["STATE"]
    elif level == "district":
        gdf, cols = load_districts(), ["STATE", "DISTRICT"]
    else:
        gdf, cols = load_subdistricts(), ["STATE", "DISTRICT", "SUB_DIST"]
    gdf = gdf[cols + ["geometry"]]
    if len(cols) == 1:
        mask = gdf[cols[0]].isin([k[0] for k in keys]).values
    else:
        mask = pd.MultiIndex.from_frame(gdf[cols]).isin(list(keys))
    sub = gdf.loc[mask]
    if sub.crs is not None and sub.crs.to_epsg() != 4326:
        sub = sub.to_crs("EPSG:4326")

    out = []
    for vals, geom in zip(sub[cols].itertuples(index=False, name=None), sub.geometry):
        if geom is None or geom.is_empty:
            continue
        try:
            geom = shapely.set_precision(geom, EXACT_PRECISION)
        except Exception:
            pass
        if geom.is_empty:
            continue
        feat = {"type": "Feature", "geometry": mapping(geom), "properties": dict(zip(cols, vals))}
        out.append((feat, tuple(vals)))
    return out


def _boundary_features(level: str, keys) -> list:
    """Full-resolution features for the selected units (falls back to the simplified boundaries
    for very large selections or on any error)."""
    keys = list(set(keys))
    if len(keys) <= EXACT_MAX_FEATURES:
        try:
            return _exact_features_cached(level, tuple(sorted(keys)))
        except Exception:
            pass
    return _simplified_features(level, keys)


def _boundary_collection(level: str, keys) -> dict:
    return {"type": "FeatureCollection", "features": [f for f, _ in _boundary_features(level, keys)]}


def _scope_frame(df: pd.DataFrame, sel: dict) -> pd.DataFrame:
    """Rows of `df` belonging to the selected units, plus an ENTITY display-name column.
    Applied identically to the Actual and Normal frames so ENTITY names match exactly."""
    level = sel["level"]
    if level == "state":
        df = df[df["STATE"].isin(sel["states"])].copy()
        df["ENTITY"] = df["STATE"]
    elif level == "district":
        keys = pd.MultiIndex.from_frame(df[["STATE", "DISTRICT"]])
        df = df[keys.isin(sel["districts"])].copy()
        multi_state = len({s for s, _ in sel["districts"]}) > 1
        df["ENTITY"] = (df["DISTRICT"] + " (" + df["STATE"] + ")") if multi_state else df["DISTRICT"]
    else:
        keys = pd.MultiIndex.from_frame(df[["STATE", "DISTRICT", "SUB_DIST"]])
        df = df[keys.isin(sel["subs"])].copy()
        multi_state = len({s for s, _, _ in sel["subs"]}) > 1
        multi_dist = len({(s, d) for s, d, _ in sel["subs"]}) > 1
        if multi_state:
            df["ENTITY"] = df["SUB_DIST"] + " (" + df["DISTRICT"] + ", " + df["STATE"] + ")"
        elif multi_dist:
            df["ENTITY"] = df["SUB_DIST"] + " (" + df["DISTRICT"] + ")"
        else:
            df["ENTITY"] = df["SUB_DIST"]
    return df


def build_rain_figure(series_list, mode: str, year: int, title: str) -> go.Figure:
    """series_list: list of dicts {name_suffix, label, data (output of aggregate_rainfall), dash}.
    'Actual' is drawn solid; 'Normal' is drawn dotted in the SAME colour for the same area."""
    fig = go.Figure()
    palette = pc.qualitative.Dark24 + pc.qualitative.Alphabet
    entities = sorted({e for s in series_list for e in s["data"]["entity"].unique()})
    color_for = {e: palette[i % len(palette)] for i, e in enumerate(entities)}

    all_periods = sorted({p for s in series_list for p in s["data"]["period"].unique()})
    x_labels = [period_label(mode, p, year) for p in all_periods]

    # Only one period selected (same start & end): a line needs >= 2 points, so show a
    # horizontal bar chart instead, one bar per area, sorted so the wettest is on top.
    if len(all_periods) == 1:
        lbl = x_labels[0].strftime("%d %b %Y") if mode == "daily" else x_labels[0]
        primary = series_list[0]["data"].set_index("entity")["rain_mm"]
        order = primary.sort_values(ascending=True, na_position="first").index.tolist()
        for i, s in enumerate(series_list):
            vals = s["data"].set_index("entity")["rain_mm"].reindex(order)
            fig.add_trace(go.Bar(
                y=order,
                x=vals.tolist(),
                orientation="h",
                name=s.get("label", "Actual"),
                marker_color=[color_for.get(e, "#888") for e in order],
                opacity=1.0 if i == 0 else 0.5,   # Normal drawn lighter
                text=[None if pd.isna(v) else f"{v:.1f}" for v in vals],
                textposition="outside",
                hovertemplate="%{y}<br>%{x:.2f} mm<extra>" + s.get("label", "Actual") + "</extra>",
            ))
        fig.update_layout(
            title=title,
            barmode="group",
            xaxis=dict(title=f"{'Rainfall' if mode == 'daily' else 'Cumulative rainfall'} (mm) — {lbl}",
                       rangemode="tozero"),
            yaxis=dict(type="category", categoryorder="array", categoryarray=order, automargin=True),
            showlegend=len(series_list) > 1,
            height=max(320, 26 * len(order) * len(series_list) + 150),
            margin=dict(l=10, r=30, t=60, b=10),
        )
        return fig

    for s in series_list:
        for entity, g in s["data"].groupby("entity"):
            g = g.set_index("period").reindex(all_periods)
            suffix = s.get("name_suffix", "")
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=g["rain_mm"].tolist(),
                mode="lines",
                name=f"{entity}{suffix}",
                legendgroup=entity,
                line=dict(color=color_for[entity], width=2, shape="spline",
                          smoothing=1.0, dash=s.get("dash", "solid")),
                connectgaps=False,
                hovertemplate="%{x}<br>%{y:.2f} mm<extra>" + f"{entity}{suffix}" + "</extra>",
            ))

    xaxis = dict(title={"daily": "Date", "weekly": "Week", "monthly": "Month"}[mode])
    if mode != "daily":
        xaxis.update(type="category", categoryorder="array", categoryarray=x_labels, tickangle=-45)
    fig.update_layout(
        title=title,
        xaxis=xaxis,
        yaxis=dict(title="Rainfall (mm)" if mode == "daily" else "Cumulative rainfall (mm)", rangemode="tozero"),
        legend=dict(title="", orientation="v"),
        hovermode="x unified" if len(entities) <= 8 else "closest",
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def build_rain_result(mode, r_start, r_end, year, sel: dict) -> dict:
    """Aggregate Actual and Normal rainfall for the SELECTED units, classify the deviation and
    prepare the map layer. Called when a Show button is clicked; the returned plain dict is stored
    in session_state, so later reruns just display it. The graph itself is built lazily
    (Plot Graph button) from result['series'].

    Only the deepest selected level is used (no automatic expansion to child units)."""
    level = sel["level"]
    if level == "india":
        return {"status": "info", "msg": "Select at least one State (and optionally Districts / SubDistricts) "
                                         "above the map, then click the Show button again."}

    note = None
    if level == "state":
        df_a, df_n = load_state_rainfall(), load_state_normal(year)
        id_cols = ["STATE"]
        scope_label = summarize_names(sel["states"])
        note = ("State rainfall = area-weighted mean of the daily mean rainfall of its districts "
                "(derived from the district files).")
    elif level == "district":
        df_a, df_n = load_rainfall("district"), load_rainfall_normal("district", year)
        id_cols = ["STATE", "DISTRICT"]
        scope_label = summarize_names([d for _, d in sel["districts"]])
    else:
        df_a, df_n = load_rainfall("subdistrict"), load_rainfall_normal("subdistrict", year)
        id_cols = ["STATE", "DISTRICT", "SUB_DIST"]
        scope_label = summarize_names([s for _, _, s in sel["subs"]])

    df_a = _scope_frame(df_a, sel)
    df_n = _scope_frame(df_n, sel)

    if df_a.empty:
        return {"status": "warning", "msg": "No rainfall records found for the selected area(s). "
                                            "Check that area names in the data match the boundary names."}

    # Like-for-like comparison: only use normal days up to the last day with actual data.
    last_date = df_a["date"].max()
    df_n = df_n[df_n["date"] <= last_date]

    agg_a = aggregate_rainfall(df_a, "ENTITY", mode, r_start, r_end)
    if agg_a.empty or agg_a["rain_mm"].isna().all():
        return {"status": "warning", "msg": "No rainfall data in the selected range."}

    normal_ok = not df_n.empty
    if normal_ok:
        agg_n = aggregate_rainfall(df_n, "ENTITY", mode, r_start, r_end)
    else:
        agg_n = pd.DataFrame(columns=["entity", "period", "rain_mm"])
        note = ((note + " ") if note else "") + (
            "No normal-rainfall records matched the selected area(s); deviation is shown as 'No Data'.")

    if mode == "daily":
        range_txt = f"{r_start:%d %b %Y} – {r_end:%d %b %Y}"
    elif mode == "weekly":
        range_txt = f"{week_label(year, r_start)} to {week_label(year, r_end)}"
    else:
        range_txt = f"{month_label(r_start)} to {month_label(r_end)} {year}"
    mode_txt = {"daily": "Daily", "weekly": "Weekly cumulative", "monthly": "Monthly cumulative"}[mode]

    # ---- deviation over the whole selected range, per area ----
    tot_a = agg_a.groupby("entity")["rain_mm"].sum(min_count=1)
    tot_n = agg_n.groupby("entity")["rain_mm"].sum(min_count=1) if normal_ok else pd.Series(dtype=float)

    ids = (df_a[id_cols + ["ENTITY"]].drop_duplicates("ENTITY")
           .rename(columns={"ENTITY": "entity"}).reset_index(drop=True))

    records = {}
    for _, r in ids.iterrows():
        ent = r["entity"]
        a, n = tot_a.get(ent, np.nan), tot_n.get(ent, np.nan)
        cat, dev = classify_deviation(a, n)
        records[tuple(r[c] for c in id_cols)] = {"entity": ent, "actual": a, "normal": n, "dev": dev, "cat": cat}

    feats = []
    for f, key in _boundary_features(level, records.keys()):
        rec = records[key]
        feats.append({
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {
                "name": rec["entity"],
                "actual": _fmt_mm(rec["actual"]),
                "normal": _fmt_mm(rec["normal"]),
                "dev": _fmt_dev(rec["dev"]),
                "category": rec["cat"],
                "color": DEV_COLOR[rec["cat"]],
            },
        })
    geojson = {"type": "FeatureCollection", "features": feats}
    if not feats:
        note = ((note + " ") if note else "") + "Could not match the selected areas to boundary polygons for the map."

    # ---- CSV export of exactly the plotted data (per period) ----
    out = agg_a.merge(agg_n.rename(columns={"rain_mm": "normal_mm"}), on=["entity", "period"], how="left")
    if "normal_mm" not in out.columns:
        out["normal_mm"] = np.nan
    out = out.merge(ids, on="entity", how="left")
    fmt = "%d-%m-%Y"
    starts, ends = [], []
    for p in out["period"]:
        if mode == "daily":
            d0 = d1 = date(year, 1, 1) + timedelta(days=int(p) - 1)
        elif mode == "weekly":
            d0, d1 = week_bounds(year, int(p))
        else:
            d0, d1 = month_bounds(year, int(p))
        starts.append(d0.strftime(fmt))
        ends.append(d1.strftime(fmt))
    cls = [classify_deviation(a, n) for a, n in zip(out["rain_mm"], out["normal_mm"])]
    export = out[id_cols].copy()
    export["Aggregation"] = mode_txt
    export["Period"] = [period_label(mode, p, year).strftime(fmt) if mode == "daily"
                        else period_label(mode, p, year) for p in out["period"]]
    export["Period_Start"] = starts
    export["Period_End"] = ends
    export["Rainfall_mm"] = out["rain_mm"].round(3)
    export["Normal_Rainfall_mm"] = out["normal_mm"].round(3)
    export["Deviation_pct"] = [None if pd.isna(c[1]) else round(c[1], 1) for c in cls]
    export["Category"] = [c[0] for c in cls]

    series_list = [{"name_suffix": " (Actual)", "label": "Actual", "data": agg_a, "dash": "solid"}]
    if normal_ok:
        series_list.append({"name_suffix": " (Normal)", "label": "Normal", "data": agg_n, "dash": "dot"})

    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", "_".join(selection_names(sel)[:3]))
    return {
        "status": "ok",
        "sel": sel,                       # selection this result was computed for (map uses it)
        "mode": mode,
        "year": year,
        "scope": scope_label,
        "range_txt": range_txt,
        "mode_txt": mode_txt,
        "geojson": geojson,
        "series": series_list,
        "fig_title": f"{mode_txt} rainfall: Actual vs Normal — {scope_label}<br><sup>{range_txt}</sup>",
        "fig": None,                      # filled when "Plot Graph" is clicked
        "note": note,
        "csv": export.to_csv(index=False).encode("utf-8"),
        "file": f"Rainfall_{mode}_{level}_{tag}.csv",
    }


@st.cache_resource(show_spinner="Loading all datasets (first start only)...")
def warm_up_all() -> bool:
    """Touch every cached loader once per server process so later interactions are served
    from memory. MODIS rasters are NOT preloaded (they are fetched per fortnight on demand)."""
    load_states()
    load_districts()
    load_subdistricts()
    state_list()
    district_lookup()
    subdistrict_lookup()
    state_bounds_lookup()
    district_bounds_lookup()
    subdistrict_bounds_lookup()
    india_states_geojson_simplified()
    districts_geojson_simplified_by_state()
    subdistricts_geojson_simplified_by_district()
    state_labels()
    district_labels_by_state()
    subdistrict_labels_by_district()
    actual_d = load_rainfall("district")
    load_rainfall("subdistrict")
    yr = int(actual_d["date"].max().year)
    load_rainfall_normal("district", yr)
    load_rainfall_normal("subdistrict", yr)
    return True


# --------------------------------------------------------------------------------------
# MAP HELPERS
# --------------------------------------------------------------------------------------
def _subset_geojson(geojson: dict, predicate) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [f for f in geojson.get("features", []) if predicate(f.get("properties", {}))],
    }


def add_label_layer(m: folium.Map, items: list, layer_name: str, font_px: int = 10, color: str = "#ffffff"):
    """Always-visible text labels (no hover), centred on the given (text, lat, lon) points.
    The inner div is absolutely positioned so translate(-50%,-50%) uses the TEXT's own width
    (inside a 0x0 DivIcon a normal block would have zero width and the text would start at the
    point and run to the right)."""
    fg = folium.FeatureGroup(name=layer_name)
    shadow = "-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000,0 0 3px #000"
    for text, lat, lon in items:
        label_html = (
            f'<div style="position:absolute;left:0;top:0;transform:translate(-50%,-50%);'
            f'white-space:nowrap;pointer-events:none;text-align:center;'
            f'font:700 {font_px}px/1 sans-serif;color:{color};text-shadow:{shadow};">'
            f"{_html.escape(str(text))}</div>"
        )
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=label_html, icon_size=(0, 0), icon_anchor=(0, 0)),
        ).add_to(fg)
    fg.add_to(m)


def rain_label_items(sel: dict) -> list:
    """Label points for exactly the units that are coloured on the rainfall map."""
    level = sel["level"]
    if level == "state":
        chosen = set(sel["states"])
        return [t for t in state_labels() if t[0] in chosen]
    if level == "district":
        dl = district_labels_by_state()
        return [t for (s, d) in sel["districts"] for t in dl.get(s, []) if t[0] == d]
    sl = subdistrict_labels_by_district()
    return [t for (s, d, sd) in sel["subs"] for t in sl.get((s, d), []) if t[0] == sd]


def add_rain_layer(m: folium.Map, geojson: dict):
    folium.GeoJson(
        geojson,
        name="Rainfall deviation from normal",
        style_function=lambda f: {
            "color": "#222222", "weight": 1.2,
            "fillColor": f["properties"]["color"], "fillOpacity": 0.8,
        },
        highlight_function=lambda f: {"weight": 3, "color": "#000000"},
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "actual", "normal", "dev", "category"],
            aliases=["Area", "Actual", "Normal", "Deviation", "Category"],
            sticky=True,
        ),
    ).add_to(m)
    m.get_root().html.add_child(folium.Element(rain_legend_html()))


def build_map(sel: dict, kind: str = "ndvi", rain_geojson: dict = None) -> folium.Map:
    """kind='ndvi' : boundary map with the NDVI overlay (selected units highlighted in red/orange)
       kind='rain' : boundary map with the rainfall-deviation choropleth for `rain_geojson`"""
    level = sel["level"]
    is_rain = kind == "rain"
    m = folium.Map(location=INDIA_CENTER, zoom_start=INDIA_ZOOM, tiles=None)

    # ---- basemaps (free, no API key) ----
    folium.TileLayer(
        tiles=ESRI_IMAGERY_URL, attr=ESRI_IMAGERY_ATTR,
        name="Satellite (Esri World Imagery)", max_zoom=19,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)

    # ---- state outlines (always) ----
    states_gj = india_states_geojson_simplified()
    folium.GeoJson(
        states_gj, name="States",
        style_function=lambda x: {"color": "#ffffff", "weight": 1.5, "fillOpacity": 0},
    ).add_to(m)

    if not is_rain:
        if level == "state":
            folium.GeoJson(
                _boundary_collection("state", [(s,) for s in sel["states"]]),
                name="Selected States",
                style_function=lambda x: {"color": "#ff3b30", "weight": 3, "fillColor": "#ff3b30", "fillOpacity": 0.08},
            ).add_to(m)

        # ---- districts of the selected state(s) ----
        if sel["states"]:
            dist_gj_by_state = districts_geojson_simplified_by_state()
            for s in sel["states"]:
                gj = dist_gj_by_state.get(s)
                if gj:
                    folium.GeoJson(
                        gj, name=f"Districts — {s}",
                        style_function=lambda x: {"color": "#0022ff", "weight": 1.2, "fillOpacity": 0},
                    ).add_to(m)
            if level == "district":
                sub = _boundary_collection("district", sel["districts"])
                if sub["features"]:
                    folium.GeoJson(
                        sub, name="Selected Districts",
                        style_function=lambda x: {"color": "#ff3b30", "weight": 3,
                                                  "fillColor": "#ff3b30", "fillOpacity": 0.08},
                    ).add_to(m)

        # ---- subdistricts of the selected district(s) ----
        if sel["districts"]:
            sub_gj_by_dist = subdistricts_geojson_simplified_by_district()
            for key in sel["districts"]:
                gj = sub_gj_by_dist.get(key)
                if gj:
                    folium.GeoJson(
                        gj, name=f"SubDistricts — {key[1]}",
                        style_function=lambda x: {"color": "#680e30", "weight": 1.2, "fillOpacity": 0},
                    ).add_to(m)
            if level == "subdistrict":
                sub = _boundary_collection("subdistrict", sel["subs"])
                if sub["features"]:
                    folium.GeoJson(
                        sub, name="Selected SubDistricts",
                        style_function=lambda x: {"color": "#ff9500", "weight": 3,
                                                  "fillColor": "#ff9500", "fillOpacity": 0.1},
                    ).add_to(m)

        # ---- NDVI overlay ----
        if st.session_state.ndvi_array is not None:
            png_b64 = array_to_png_base64(st.session_state.ndvi_array)
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{png_b64}",
                bounds=st.session_state.ndvi_bounds,
                opacity=0.85,
                name=st.session_state.ndvi_label,
            ).add_to(m)
            m.get_root().html.add_child(folium.Element(ndvi_legend_html()))

        # ---- always-on labels for the CURRENT vector layer ----
        if not sel["states"]:
            add_label_layer(m, state_labels(), "Labels — States", font_px=12)
        elif not sel["districts"]:
            items = []
            dl = district_labels_by_state()
            for s in sel["states"]:
                items.extend(dl.get(s, []))
            add_label_layer(m, items, "Labels — Districts", font_px=10)
        else:
            items = []
            sl = subdistrict_labels_by_district()
            for key in sel["districts"]:
                items.extend(sl.get(key, []))
            add_label_layer(m, items, "Labels — SubDistricts", font_px=9)
    else:
        # ---- rainfall deviation choropleth + labels for the coloured units ----
        if rain_geojson is not None and rain_geojson.get("features"):
            add_rain_layer(m, rain_geojson)
        font_px = {"state": 12, "district": 10, "subdistrict": 9}.get(level, 10)
        add_label_layer(m, rain_label_items(sel), "Labels", font_px=font_px)

    folium.LayerControl().add_to(m)

    # ---- zoom to the selection ----
    b = selection_bounds(sel)
    if b is not None:
        m.fit_bounds(b)
    return m


# --------------------------------------------------------------------------------------
# FILTERS (main window, horizontal)
# --------------------------------------------------------------------------------------
def reset_filters():
    """Back to the homepage: All States, no NDVI overlay, no rainfall result."""
    for k in FLT_KEYS:
        st.session_state[k] = []
    st.session_state.ndvi_array = None
    st.session_state.ndvi_bounds = None
    st.session_state.ndvi_label = None
    st.session_state.rain_result = None


def render_filters() -> dict:
    for k in FLT_KEYS:
        st.session_state.setdefault(k, [])

    states_all = state_list()
    st.session_state["flt_states"] = [s for s in st.session_state["flt_states"] if s in states_all]

    c1, c2, c3, c4 = st.columns([3, 3, 3, 1], vertical_alignment="bottom")

    with c1:
        sel_states = st.multiselect(
            "State", states_all, key="flt_states", placeholder="All States (India)"
        )

    dist_opts = [(s, d) for s in sel_states for d in district_lookup().get(s, [])]
    st.session_state["flt_districts"] = [x for x in st.session_state["flt_districts"] if x in dist_opts]
    multi_state = len(sel_states) > 1
    with c2:
        sel_districts = st.multiselect(
            "District", dist_opts, key="flt_districts",
            format_func=lambda x: f"{x[1]} ({x[0]})" if multi_state else x[1],
            placeholder="All Districts" if dist_opts else "Select a State first",
            disabled=not dist_opts,
        )

    sub_opts = [
        (s, d, sd)
        for (s, d) in sel_districts
        for sd in subdistrict_lookup().get((s, d), [])
    ]
    st.session_state["flt_subs"] = [x for x in st.session_state["flt_subs"] if x in sub_opts]
    multi_dist = len(sel_districts) > 1
    with c3:
        st.multiselect(
            "SubDistrict", sub_opts, key="flt_subs",
            format_func=lambda x: f"{x[2]} ({x[1]})" if multi_dist else x[2],
            placeholder="All SubDistricts" if sub_opts else "Select a District first",
            disabled=not sub_opts,
        )

    with c4:
        st.button("Reset", key="btn_reset", on_click=reset_filters, width="stretch",
                  help="Clear all filters and go back to the All-States home view")

    return get_selection()


# --------------------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------------------
def app():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center; margin:0; padding:0;'> 🌿 Agri-Insights 🛰️</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center; font-size:0.95rem; color:#7a7a7a; margin:0.2rem 0 0.8rem 0;'>"
        f"{APP_DESCRIPTION}</p>",
        unsafe_allow_html=True,
    )
    warm_up_all()

    if "ndvi_array" not in st.session_state:
        st.session_state.ndvi_array = None
        st.session_state.ndvi_bounds = None
        st.session_state.ndvi_label = None
    if "rain_result" not in st.session_state:
        st.session_state.rain_result = None

    # ======================= MAIN WINDOW: ADMIN BOUNDARY FILTERS =======================
    sel = render_filters()
    level = sel["level"]

    # ======================= SIDEBAR 1: NDVI (collapsible) =======================
    avail = {}
    avail_error = None
    try:
        avail = list_available_ndvi()
    except Exception as e:
        avail_error = str(e)

    with st.sidebar.expander("NDVI", expanded=True):
        year_options = sorted(avail.keys()) if avail else years_list
        selected_year = st.selectbox("Select Year", year_options, index=len(year_options) - 1)

        if avail:
            fortnight_options = available_fortnights(selected_year, avail)
        else:
            fortnight_options = list(fn_dict.keys())  # listing failed -> fall back to all
            if avail_error:
                st.caption("Could not list files in the bucket; showing all fortnights.")
        if not fortnight_options:
            st.warning("No NDVI files found for this year.")
        selected_fortnight = st.selectbox(
            "Select Fortnight", fortnight_options,
            index=max(len(fortnight_options) - 1, 0) if fortnight_options else None,
        )

        load_clicked = st.button(
            "Load NDVI", type="primary",
            disabled=(level == "india" or selected_fortnight is None),
        )
        if level == "india":
            st.caption("Select at least one State to enable Load NDVI.")

    # ======================= SIDEBAR 2: RAINFALL (collapsible) =======================
    rain_ref = None
    rain_year = None
    try:
        rain_ref = load_rainfall("district")  # used only to derive filter ranges/options
        rain_year = int(rain_ref["date"].max().year)
    except Exception as e:
        st.sidebar.error(f"Could not load rainfall data: {e}")

    with st.sidebar.expander("Rainfall", expanded=True):
        if rain_ref is not None and not rain_ref.empty:
            min_date = rain_ref["date"].min().date()
            max_date = rain_ref["date"].max().date()
            ref_p = add_period_columns(rain_ref)
            weeks_avail = sorted(ref_p["week"].unique().tolist())
            months_avail = sorted(ref_p["month"].unique().tolist())
            week_labels = {week_label(rain_year, w): w for w in weeks_avail}
            month_labels = {month_label(m_): m_ for m_ in months_avail}

            st.caption(f"Rainfall data available: {min_date:%d %b %Y} – {max_date:%d %b %Y}")

            tab_d, tab_w, tab_m = st.tabs(["Daily", "Weekly", "Monthly"])

            # ---- Daily filter ----
            with tab_d:
                default_start = max(min_date, max_date - timedelta(days=29))
                daily_range = st.date_input(
                    "Date range (start – end)",
                    value=(default_start, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="rain_daily_range",
                )
                if st.button("Show daily rainfall", key="btn_rain_daily"):
                    if isinstance(daily_range, (tuple, list)) and len(daily_range) == 2:
                        st.session_state.rain_result = build_rain_result(
                            "daily", daily_range[0], daily_range[1], rain_year, sel)
                    else:
                        st.warning("Pick both a start and an end date.")

            # ---- Weekly filter ----
            with tab_w:
                wk_names = list(week_labels.keys())
                wk_start = st.selectbox("Start week", wk_names, index=0, key="rain_wk_start")
                wk_end = st.selectbox("End week", wk_names, index=len(wk_names) - 1, key="rain_wk_end")
                if st.button("Show weekly rainfall", key="btn_rain_weekly"):
                    if week_labels[wk_start] > week_labels[wk_end]:
                        st.warning("Start week must not be after end week.")
                    else:
                        st.session_state.rain_result = build_rain_result(
                            "weekly", week_labels[wk_start], week_labels[wk_end], rain_year, sel)

            # ---- Monthly filter ----
            with tab_m:
                mo_names = list(month_labels.keys())
                mo_start = st.selectbox("Start month", mo_names, index=0, key="rain_mo_start")
                mo_end = st.selectbox("End month", mo_names, index=len(mo_names) - 1, key="rain_mo_end")
                if st.button("Show monthly rainfall", key="btn_rain_monthly"):
                    if month_labels[mo_start] > month_labels[mo_end]:
                        st.warning("Start month must not be after end month.")
                    else:
                        st.session_state.rain_result = build_rain_result(
                            "monthly", month_labels[mo_start], month_labels[mo_end], rain_year, sel)

    # ---------------- Handle Load NDVI click ----------------
    if load_clicked and level != "india" and selected_fortnight is not None:
        try:
            clip_geom, clip_crs = selection_geometry(sel)
        except Exception as e:
            clip_geom, clip_crs = None, None
            st.error(f"Could not build the clip geometry: {e}")

        clip_label = summarize_names(selection_names(sel))

        if clip_geom is None:
            st.error("Could not find geometry for the selected area(s).")
        else:
            filename = build_raster_filename(selected_year, selected_fortnight)
            bucket_path = build_raster_bucket_path(filename)

            with st.spinner(f"Reading NDVI window for {clip_label}..."):
                try:
                    local_raster_path = get_local_raster_path(bucket_path)
                    ndvi_array, bounds_4326 = fetch_and_clip_ndvi(
                        local_raster_path, clip_geom.wkt, clip_crs
                    )
                    st.session_state.ndvi_array = ndvi_array
                    st.session_state.ndvi_bounds = bounds_4326
                    st.session_state.ndvi_label = f"NDVI — {clip_label} ({selected_fortnight}, {selected_year})"
                    st.session_state.ndvi_tag = (
                        re.sub(r"[^A-Za-z0-9_-]+", "_", "_".join(selection_names(sel)[:3]))
                        + (f"_and_{len(selection_names(sel)) - 3}_more" if len(selection_names(sel)) > 3 else "")
                    )
                    st.success("NDVI loaded.")
                except Exception as e:
                    st.session_state.ndvi_array = None
                    st.error(
                        f"Could not load raster for this fortnight/year (file: {filename}). "
                        f"It may not exist yet on Hugging Face. Details: {e}"
                    )

    # ---------------- NDVI map (collapsible) ----------------
    m = build_map(sel, kind="ndvi")
    with st.expander("NDVI", expanded=True):
        # returned_objects=[] -> panning/zooming the map does not trigger a Streamlit rerun
        st_folium(m, width='stretch', height=650, key="ndvi_map", returned_objects=[])

    # ---------------- Rainfall: deviation map -> Plot Graph button -> graph (collapsible) ----------------
    # Everything heavy is computed once, when a "Show ... rainfall" button is clicked
    # (build_rain_result). The graph is built only when "Plot Graph" is clicked. Later reruns
    # (switching boundaries, loading NDVI, ...) just display the stored result.
    rain_res = st.session_state.rain_result
    if rain_res is not None:
        with st.expander("Rainfall", expanded=True):
            if rain_res["status"] == "ok":
                st.markdown(
                    f"**Rainfall deviation from normal — {rain_res['scope']}**  \n"
                    f"{rain_res['mode_txt']} · {rain_res['range_txt']}"
                )
                rain_map = build_map(rain_res["sel"], kind="rain", rain_geojson=rain_res["geojson"])
                st_folium(rain_map, width='stretch', height=600, key="rain_map", returned_objects=[])

                if st.button("Plot Graph", key="btn_plot_graph", type="primary"):
                    rain_res["fig"] = build_rain_figure(
                        rain_res["series"], rain_res["mode"], rain_res["year"], rain_res["fig_title"])
                if rain_res.get("fig") is not None:
                    st.plotly_chart(rain_res["fig"], width='stretch')

                # if rain_res.get("note"):
                #     st.caption(rain_res["note"])
                # st.caption(
                #     "Deviation = (Actual − Normal) / Normal × 100, over the selected period; normal is summed "
                #     "only for days up to the latest available actual data."
                # )
            elif rain_res["status"] == "info":
                st.info(rain_res["msg"])
            else:
                st.warning(rain_res["msg"])

    # ---------------- Download ----------------
    ndvi_ready = st.session_state.ndvi_array is not None
    rain_ready = rain_res is not None and rain_res["status"] == "ok"
    if ndvi_ready or rain_ready:
        st.subheader("Download")
        if ndvi_ready:
            tif_bytes = clipped_array_to_geotiff_bytes(
                st.session_state.ndvi_array, st.session_state.ndvi_bounds
            )
            area_tag = st.session_state.get("ndvi_tag") or "selection"
            st.download_button(
                label="Download NDVI (GeoTIFF)",
                data=tif_bytes,
                file_name=f"NDVI_{area_tag}_{selected_year}_{(selected_fortnight or '').replace(' ', '')}.tif",
                mime="image/tiff",
            )
        if rain_ready:
            st.download_button(
                label="Download rainfall data (CSV)",
                data=rain_res["csv"],
                file_name=rain_res["file"],
                mime="text/csv",
            )
        # NOTE: zonal statistics (exactextract) intentionally deferred.

    # ---------------- Citations (end of page) ----------------
    st.markdown(citation_html(), unsafe_allow_html=True)


if __name__ == "__main__":
    app()
