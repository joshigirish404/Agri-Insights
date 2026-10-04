"""
Streamlit Agri-Insights
--------------------------------------
Loads State/District/SubDistrict boundaries (GeoParquet), MODIS NDVI COGs (GeoTIFF)
and daily rainfall CSVs from a Hugging Face Storage Bucket ("girishhf07/agri_datasets").

Data layout expected in the HF Storage Bucket:
    boundaries/India_States.parquet
    boundaries/India_Districts.geoparquet
    boundaries/India_SubDistricts.geoparquet
    MOD13Q1_2026/MOD13Q1.061__250m_16_days_NDVI_doy2026<DDD>000000_aid0001.tif
    xls/district_rainfall_stats_2026.csv      (STATE, STATE_LGD, DISTRICT, DIST_LGD, date, Mean_Rainfall (mm))
    xls/subdistrict_rainfall_stats_2026.csv   (STATE, STATE_LGD, DISTRICT, DIST_LGD, SUB_DIST, date, Mean_Rainfall (mm))

Sidebar sections:
    1. Admin Boundary  (State -> District -> SubDistrict)
    2. NDVI            (Year, Fortnight, Load NDVI)            <- NDVI workflow unchanged
    3. Rainfall        (Daily range / Week range / Month range -> smoothed line graph)

Rainfall design notes:
    * Only DAILY rainfall is stored. Weekly and monthly values are cumulative sums of daily data.
    * Weeks are fixed 7-day blocks counted from 01 Jan (Week 1 = 01 Jan - 07 Jan, Week 2 = 08 Jan - 14 Jan ...).
    * Aggregation is keyed on integer period keys (day-of-year / week no. / month no.) so a
      "Normal rainfall" series can later be aggregated the same way and overlaid on the same x-axis.
    * The graph follows the CURRENT admin selection:
        State + All Districts          -> one line per district of the state
        District + All SubDistricts    -> one line per subdistrict of the district
        SubDistrict selected           -> single line
        All States                     -> prompt to pick a state
"""

import base64
import calendar
import io
import os
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
import rasterio
import streamlit as st
from huggingface_hub import download_bucket_files
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
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

DISTRICT_RAIN_PATH = "xls/district_rainfall_stats_2026.csv"
SUBDISTRICT_RAIN_PATH = "xls/subdistrict_rainfall_stats_2026.csv"
RAIN_COL = "Mean_Rainfall (mm)"
RAIN_CACHE_MAX_AGE_S = 6 * 3600  # rainfall CSVs grow daily -> re-download if local copy is older than this

CACHE_DIR = os.path.join(tempfile.gettempdir(), "modis_ndvi_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = None

RASTER_NODATA_FILL = -3000  # MODIS NDVI fill value
NDVI_SCALE = 0.0001
NDVI_VALID_MIN, NDVI_VALID_MAX = -0.2, 1.0  # after scaling

years_list = [2026]

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

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "VSI_CACHE": "TRUE",
}


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
    max_age_s. Used for rainfall CSVs, which are appended to as new days arrive."""
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


# --------------------------------------------------------------------------------------
# RAINFALL: loading, period helpers, aggregation, plotting
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, ttl=RAIN_CACHE_MAX_AGE_S)
def load_rainfall(level: str) -> pd.DataFrame:
    """level = 'district' or 'subdistrict'. Returns a tidy daily DataFrame with columns:
    STATE, DISTRICT, [SUB_DIST], date (datetime64), rain_mm (float)."""
    bucket_path = DISTRICT_RAIN_PATH if level == "district" else SUBDISTRICT_RAIN_PATH
    path = _download_bucket_file_fresh(bucket_path)
    # sep=None lets pandas sniff comma vs tab delimiters
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    # tolerate case differences in column names
    lower_map = {c.lower(): c for c in df.columns}
    date_col = lower_map.get("date")
    rain_src = RAIN_COL if RAIN_COL in df.columns else next(
        (c for c in df.columns if "rain" in c.lower()), None
    )
    if date_col is None or rain_src is None:
        raise ValueError(f"Expected 'date' and rainfall columns; found {list(df.columns)}")

    keep = ["STATE", "DISTRICT"] + (["SUB_DIST"] if level == "subdistrict" else [])
    for c in keep:
        df[c] = df[c].astype(str).str.strip()

    df["date"] = _parse_dates(df[date_col])
    df["rain_mm"] = pd.to_numeric(df[rain_src], errors="coerce")
    n_bad = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])
    if df.empty:
        raise ValueError(
            f"No valid dates parsed from column '{date_col}' ({n_bad} unparseable rows). "
            "Expected dd-mm-yyyy."
        )
    out = df[keep + ["date", "rain_mm"]].reset_index(drop=True)
    return add_period_columns(out)  # doy/week/month computed once; treat result as read-only


def _parse_dates(s: pd.Series) -> pd.Series:
    """Try several common date formats on the raw text and keep whichever parses the most
    rows (dd-mm-yyyy is expected; the others cover CSVs re-saved by Excel/other tools)."""
    s = s.astype(str).str.strip()
    best = None
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
                "%Y-%m-%d %H:%M:%S", "%d-%m-%y", "%d/%m/%y"):
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if best is None or parsed.notna().sum() > best.notna().sum():
            best = parsed
    if best.notna().sum() < len(s) * 0.5:
        loose = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if loose.notna().sum() > best.notna().sum():
            best = loose
    return best.dt.normalize()


def add_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Integer period keys. Keying on day-of-year / week / month number (not on dates)
    means a future 'Normal rainfall' dataset can be aggregated identically and joined
    on the same keys for same-period comparison."""
    if "doy" in df.columns:  # already computed once at load time
        return df
    out = df.copy()
    out["doy"] = out["date"].dt.dayofyear
    # Week 1 = day 1-7, Week 2 = day 8-14, ... ; only 52 weeks: the leftover day(s)
    # after day 364 (31 Dec, plus 31 Dec/leap day in leap years) belong to Week 52.
    out["week"] = np.minimum((out["doy"] - 1) // 7 + 1, 52)
    out["month"] = out["date"].dt.month
    return out


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


def build_rain_figure(series_list, mode: str, year: int, title: str) -> go.Figure:
    """series_list: list of dicts {name_suffix, data (output of aggregate_rainfall), dash}.
    Today only the 'Actual' series is passed; a Normal-rainfall series can be appended later
    (same aggregation, dash='dash') and will line up on the same period keys."""
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
                opacity=1.0 if i == 0 else 0.5,   # later series (e.g. Normal) drawn lighter
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


def month_bounds(year: int, month_no: int):
    return date(year, month_no, 1), date(year, month_no, calendar.monthrange(year, month_no)[1])


def build_rain_result(mode, r_start, r_end, year, state, district, subdistrict) -> dict:
    """Aggregate + plot rainfall for the admin selection AT THE MOMENT a Show button is clicked.
    Returns a plain dict stored in session_state, so later reruns (switching boundary, loading
    NDVI, ...) just display it without recomputing anything."""
    if state is None:
        return {"status": "info", "msg": "Select a State (and optionally a District) in the sidebar, "
                                         "then click the Show button again. All-India plotting would mean hundreds of lines."}
    if district is None:
        level, entity_col = "district", "DISTRICT"
        df = load_rainfall("district")
        df = df[df["STATE"] == state]
        scope_label = f"Districts of {state}"
        scope_tag = state
    else:
        level, entity_col = "subdistrict", "SUB_DIST"
        df = load_rainfall("subdistrict")
        df = df[(df["STATE"] == state) & (df["DISTRICT"] == district)]
        if subdistrict is None:
            scope_label = f"SubDistricts of {district}, {state}"
            scope_tag = district
        else:
            df = df[df["SUB_DIST"] == subdistrict]
            scope_label = f"{subdistrict}, {district}, {state}"
            scope_tag = subdistrict

    if df.empty:
        return {"status": "warning", "msg": "No rainfall records found for the selected area. "
                                            "Check that area names in the CSV match the boundary names."}
    agg = aggregate_rainfall(df, entity_col, mode, r_start, r_end)
    if agg.empty or agg["rain_mm"].isna().all():
        return {"status": "warning", "msg": "No rainfall data in the selected range."}

    if mode == "daily":
        range_txt = f"{r_start:%d %b %Y} – {r_end:%d %b %Y}"
    elif mode == "weekly":
        range_txt = f"{week_label(year, r_start)} to {week_label(year, r_end)}"
    else:
        range_txt = f"{month_label(r_start)} to {month_label(r_end)} {year}"
    mode_txt = {"daily": "Daily", "weekly": "Weekly cumulative", "monthly": "Monthly cumulative"}[mode]

    # Normal rainfall: later, build another aggregate_rainfall(...) result from the
    # normal-rainfall dataframe and append
    # {"name_suffix": " (Normal)", "label": "Normal", "data": normal_agg, "dash": "dash"} here.
    series_list = [{"name_suffix": "", "label": "Actual", "data": agg, "dash": "solid"}]
    fig = build_rain_figure(
        series_list, mode, year,
        title=f"{mode_txt} rainfall — {scope_label}<br><sup>{range_txt}</sup>",
    )

    # ---- CSV export of exactly the plotted data ----
    out = agg.copy()
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
    export = pd.DataFrame(index=out.index)
    export["STATE"] = state
    if level == "district":
        export["DISTRICT"] = out["entity"]
    else:
        export["DISTRICT"] = district
        export["SUB_DIST"] = out["entity"]
    export["Aggregation"] = mode_txt
    export["Period"] = [period_label(mode, p, year).strftime(fmt) if mode == "daily"
                        else period_label(mode, p, year) for p in out["period"]]
    export["Period_Start"] = starts
    export["Period_End"] = ends
    export["Rainfall_mm"] = out["rain_mm"].round(3)

    tag = str(scope_tag).replace(" ", "_").replace("/", "-")
    return {
        "status": "ok",
        "fig": fig,
        "csv": export.to_csv(index=False).encode("utf-8"),
        "file": f"Rainfall_{mode}_{tag}.csv",
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
    load_rainfall("district")
    load_rainfall("subdistrict")
    return True


SIDEBAR_LABEL_CSS = """
<style>
section[data-testid="stSidebar"] label p {
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}
</style>
"""


# --------------------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------------------
def app():
    st.write("HF token configured:", bool(HF_TOKEN))

    st.markdown(SIDEBAR_LABEL_CSS, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'> 🌿 Agri-Insights 🛰️</h1>", unsafe_allow_html=True)
    warm_up_all()

    if "ndvi_array" not in st.session_state:
        st.session_state.ndvi_array = None
        st.session_state.ndvi_bounds = None
        st.session_state.ndvi_label = None
    if "rain_result" not in st.session_state:
        st.session_state.rain_result = None

    ALL_STATES_LABEL = "-- All States (India) --"
    ALL_DISTRICTS_LABEL = "-- All Districts --"
    ALL_SUBDISTRICTS_LABEL = "-- All SubDistricts --"

    # ======================= SIDEBAR 1: ADMIN BOUNDARY =======================
    # st.sidebar.header("1. Admin Boundary")
    state_options = [ALL_STATES_LABEL] + state_list()
    selected_state_raw = st.sidebar.selectbox("State", state_options, index=0)
    selected_state = None if selected_state_raw == ALL_STATES_LABEL else selected_state_raw

    if selected_state:
        district_options = [ALL_DISTRICTS_LABEL] + district_lookup().get(selected_state, [])
    else:
        district_options = [ALL_DISTRICTS_LABEL]
    selected_district_raw = st.sidebar.selectbox("District", district_options, index=0)
    selected_district = None if selected_district_raw == ALL_DISTRICTS_LABEL else selected_district_raw

    if selected_state and selected_district:
        subdistrict_options = [ALL_SUBDISTRICTS_LABEL] + subdistrict_lookup().get(
            (selected_state, selected_district), []
        )
    else:
        subdistrict_options = [ALL_SUBDISTRICTS_LABEL]
    selected_subdistrict_raw = st.sidebar.selectbox("SubDistrict", subdistrict_options, index=0)
    selected_subdistrict = None if selected_subdistrict_raw == ALL_SUBDISTRICTS_LABEL else selected_subdistrict_raw

    # ======================= SIDEBAR 2: NDVI =======================
    st.sidebar.divider()
    st.sidebar.header("NDVI")
    selected_year = st.sidebar.selectbox("Select Year", years_list)
    selected_fortnight = st.sidebar.selectbox("Select Fortnight", list(fn_dict.keys()))

    load_clicked = st.sidebar.button(
        "Load NDVI", type="primary", disabled=(selected_district is None)
    )
    if selected_district is None:
        st.sidebar.caption("Select a State and District to enable Load NDVI.")

    # ======================= SIDEBAR 3: RAINFALL =======================
    st.sidebar.divider()
    st.sidebar.header("Rainfall")

    rain_ref = None
    rain_year = None
    try:
        rain_ref = load_rainfall("district")  # used only to derive filter ranges/options
        rain_year = int(rain_ref["date"].max().year)
    except Exception as e:
        st.sidebar.error(f"Could not load rainfall data: {e}")



    if rain_ref is not None and not rain_ref.empty:
        min_date = rain_ref["date"].min().date()
        max_date = rain_ref["date"].max().date()
        ref_p = add_period_columns(rain_ref)
        weeks_avail = sorted(ref_p["week"].unique().tolist())
        months_avail = sorted(ref_p["month"].unique().tolist())
        week_labels = {week_label(rain_year, w): w for w in weeks_avail}
        month_labels = {month_label(m): m for m in months_avail}

        st.sidebar.caption(f"Rainfall data available: {min_date:%d %b %Y} – {max_date:%d %b %Y}")

        # ---- Daily filter ----
        with st.sidebar.expander("Daily filter", expanded=False):
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
                        "daily", daily_range[0], daily_range[1], rain_year,
                        selected_state, selected_district, selected_subdistrict)
                else:
                    st.warning("Pick both a start and an end date.")

        # ---- Weekly filter ----
        with st.sidebar.expander("Week filter", expanded=False):
            wk_names = list(week_labels.keys())
            wk_start = st.selectbox("Start week", wk_names, index=0, key="rain_wk_start")
            wk_end = st.selectbox("End week", wk_names, index=len(wk_names) - 1, key="rain_wk_end")
            if st.button("Show weekly rainfall", key="btn_rain_weekly"):
                if week_labels[wk_start] > week_labels[wk_end]:
                    st.warning("Start week must not be after end week.")
                else:
                    st.session_state.rain_result = build_rain_result(
                        "weekly", week_labels[wk_start], week_labels[wk_end], rain_year,
                        selected_state, selected_district, selected_subdistrict)

        # ---- Monthly filter ----
        with st.sidebar.expander("Month filter", expanded=False):
            mo_names = list(month_labels.keys())
            mo_start = st.selectbox("Start month", mo_names, index=0, key="rain_mo_start")
            mo_end = st.selectbox("End month", mo_names, index=len(mo_names) - 1, key="rain_mo_end")
            if st.button("Show monthly rainfall", key="btn_rain_monthly"):
                if month_labels[mo_start] > month_labels[mo_end]:
                    st.warning("Start month must not be after end month.")
                else:
                    st.session_state.rain_result = build_rain_result(
                        "monthly", month_labels[mo_start], month_labels[mo_end], rain_year,
                        selected_state, selected_district, selected_subdistrict)

    # ---------------- Handle Load NDVI click (unchanged) ----------------
    if load_clicked and selected_district:
        clip_geom = None
        clip_crs = None
        clip_label = None

        if selected_subdistrict:
            subdistricts_gdf = load_subdistricts()
            row = subdistricts_gdf.loc[
                (subdistricts_gdf["STATE"] == selected_state)
                & (subdistricts_gdf["DISTRICT"] == selected_district)
                & (subdistricts_gdf["SUB_DIST"] == selected_subdistrict)
            ]
            if not row.empty:
                clip_geom = row.geometry.iloc[0]
                clip_crs = str(row.crs)
                clip_label = f"{selected_subdistrict}, {selected_district}, {selected_state}"
        else:
            districts_gdf = load_districts()
            row = districts_gdf.loc[
                (districts_gdf["STATE"] == selected_state) & (districts_gdf["DISTRICT"] == selected_district)
            ]
            if not row.empty:
                clip_geom = row.geometry.iloc[0]
                clip_crs = str(row.crs)
                clip_label = f"{selected_district}, {selected_state}"

        if clip_geom is None:
            st.error("Could not find geometry for the selected area.")
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
                    st.success("NDVI loaded.")
                except Exception as e:
                    st.session_state.ndvi_array = None
                    st.error(
                        f"Could not load raster for this fortnight/year (file: {filename}). "
                        f"It may not exist yet on Hugging Face. Details: {e}"
                    )

    # ---------------- Build map ----------------
    m = folium.Map(location=INDIA_CENTER, zoom_start=INDIA_ZOOM, tiles="OpenStreetMap")

    folium.GeoJson(
        india_states_geojson_simplified(),
        name="States",
        style_function=lambda x: {"color": "#555555", "weight": 1, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(fields=["STATE"]),
    ).add_to(m)

    if selected_state:
        state_districts_geojson = districts_geojson_simplified_by_state().get(selected_state)
        if state_districts_geojson:
            folium.GeoJson(
                state_districts_geojson,
                name=f"Districts — {selected_state}",
                style_function=lambda x: {"color": "#1f78b4", "weight": 1.5, "fillOpacity": 0},
                highlight_function=lambda x: {"weight": 3},
                tooltip=folium.GeoJsonTooltip(fields=["DISTRICT"]),
            ).add_to(m)

    selected_district_feature = None
    if selected_state and selected_district:
        state_districts_geojson = districts_geojson_simplified_by_state().get(selected_state, {})
        selected_district_feature = next(
            (f for f in state_districts_geojson.get("features", [])
             if f["properties"].get("DISTRICT") == selected_district),
            None,
        )
        if selected_district_feature:
            folium.GeoJson(
                selected_district_feature,
                name="Selected District",
                style_function=lambda x: {"color": "#e31a1c", "weight": 3, "fillOpacity": 0.05, "fillColor": "#e31a1c"},
            ).add_to(m)

    if selected_state and selected_district:
        district_subdistricts_geojson = subdistricts_geojson_simplified_by_district().get(
            (selected_state, selected_district)
        )
        if district_subdistricts_geojson:
            folium.GeoJson(
                district_subdistricts_geojson,
                name=f"SubDistricts — {selected_district}",
                style_function=lambda x: {"color": "#6a3d9a", "weight": 1, "fillOpacity": 0},
                highlight_function=lambda x: {"weight": 2.5},
                tooltip=folium.GeoJsonTooltip(fields=["SUB_DIST"]),
            ).add_to(m)

            if selected_subdistrict:
                selected_subdistrict_feature = next(
                    (f for f in district_subdistricts_geojson.get("features", [])
                     if f["properties"].get("SUB_DIST") == selected_subdistrict),
                    None,
                )
                if selected_subdistrict_feature:
                    folium.GeoJson(
                        selected_subdistrict_feature,
                        name="Selected SubDistrict",
                        style_function=lambda x: {"color": "#ff7f00", "weight": 3, "fillOpacity": 0.08, "fillColor": "#ff7f00"},
                    ).add_to(m)

    if st.session_state.ndvi_array is not None:
        png_b64 = array_to_png_base64(st.session_state.ndvi_array)
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{png_b64}",
            bounds=st.session_state.ndvi_bounds,
            opacity=0.85,
            name=st.session_state.ndvi_label,
        ).add_to(m)
        m.get_root().html.add_child(folium.Element(ndvi_legend_html()))

    folium.LayerControl().add_to(m)

    # ---------------- Zoom logic ----------------
    if selected_state and selected_district and selected_subdistrict:
        b = subdistrict_bounds_lookup().get((selected_state, selected_district, selected_subdistrict))
        if b is not None:
            m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    elif selected_state and selected_district:
        b = district_bounds_lookup().get((selected_state, selected_district))
        if b is not None:
            m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    elif selected_state:
        b = state_bounds_lookup().get(selected_state)
        if b is not None:
            m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

    st.subheader("NDVI")
    st_folium(m, width='stretch', height=650, key="ndvi_map")

    # ---------------- Rainfall graph ----------------
    # Computed only when a "Show ... rainfall" button is clicked (see build_rain_result);
    # here we only display the stored result, so switching boundaries or loading NDVI
    # does not touch the rainfall data.
    rain_res = st.session_state.rain_result
    if rain_res is not None:
        st.subheader("Rainfall")
        if rain_res["status"] == "ok":
            st.plotly_chart(rain_res["fig"], width='stretch')
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
            area_tag = selected_subdistrict or selected_district
            st.download_button(
                label="Download NDVI (GeoTIFF)",
                data=tif_bytes,
                file_name=f"NDVI_{area_tag}_{selected_year}_{selected_fortnight.replace(' ', '')}.tif",
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

if __name__ == "__main__":
    app()
