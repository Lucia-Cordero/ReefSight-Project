from functools import lru_cache
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from io import StringIO
from scipy.spatial import cKDTree
import math
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import xarray as xr
from pyproj import Geod
from concurrent.futures import ThreadPoolExecutor, as_completed


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate haversine geo distances (slightly differs from 'asin' version).
    Output mathematically the same (computational reasons)
    """

    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@lru_cache(maxsize=8)
def _get_erddap_grid(dataset_id):
    """
    Retrieve and cache the latitude/longitude grid for an ERDDAP dataset.
    Detects latitude/longitude columns by substring matching for flexibility.
    Returns (tree, grid_points) where tree is a KDTree over (lat, lon).
    """
    url = f"https://coastwatch.noaa.gov/erddap/griddap/{dataset_id}.csvp?latitude,longitude"
    r = requests.get(url, timeout=30)
    df = pd.read_csv(StringIO(r.text))

    # Find latitude / longitude columns by substring match
    cols = df.columns.tolist()
    lat_col_candidates = [c for c in cols if "lat" in c.lower()]
    lon_col_candidates = [c for c in cols if "lon" in c.lower()]

    if not lat_col_candidates or not lon_col_candidates:
        raise RuntimeError(
            f"Could not identify latitude/longitude columns for {dataset_id}. "
            f"Columns are: {cols}"
        )

    lat_col = lat_col_candidates[0]
    lon_col = lon_col_candidates[0]

    # Filter out NaN/inf values and build grid
    grid_df = df[[lat_col, lon_col]].dropna()
    grid_df = grid_df[np.isfinite(grid_df[lat_col]) & np.isfinite(grid_df[lon_col])]

    if grid_df.empty:
        raise RuntimeError(f"No valid grid points found for {dataset_id}")

    grid_points = grid_df.to_numpy()
    tree = cKDTree(grid_points)
    return tree, grid_points

def _try_nearest_4d(dataset_id, variable, time_str, lat, lon, max_km=20, max_days=365):
    """
    FINAL FALLBACK: ERDDAP orderByClosest(time/lat/lon) within bounds.
    Returns (value, source_str) or (nan, None)
    """
    t0 = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")

    # Time window: ±max_days
    t_start = (t0 - timedelta(days=max_days)).strftime("%Y-%m-%d")
    t_end = (t0 + timedelta(days=max_days)).strftime("%Y-%m-%d")

    # 2° spatial window around target
    lat_min, lat_max = lat-1, lat+1
    lon_min, lon_max = lon-1, lon+1

    url = (
        f"https://coastwatch.noaa.gov/erddap/griddap/{dataset_id}.csv?"
        f"{variable}"
        f"[({t_start}):({t_end})][({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
    )

    try:
        r = requests.get(url, timeout=30)
        df = pd.read_csv(StringIO(r.text), skiprows=[1])

        if df.empty or variable not in df.columns:
            return np.nan, None

        df[variable] = pd.to_numeric(df[variable], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=[variable, 'latitude', 'longitude', 'time'])

        df['gc_dist_km'] = df.apply(
            lambda row: haversine(lat, lon, row['latitude'], row['longitude']) / 1000, axis=1
            )
        df['time_diff'] = (df['time'] - t0).abs()
        df = df.sort_values(['gc_dist_km', 'time_diff'])

        for _, row in df.iterrows():
            if row['gc_dist_km'] <= max_km:
                try:
                    row_time = pd.to_datetime(row['time']).strftime("%Y-%m-%d")
                except Exception:
                    row_time = str(row['time'])[:10]
                return float(row[variable]), f"nearest_4d({row['gc_dist_km']:.1f}km,{row_time})"

        return np.nan, None
    except:
        return np.nan, None


@lru_cache(maxsize=256)
def erddap_extract(dataset_id, variable, time_str, lat, lon, max_days_back=7, max_km=50):
    """
    Extract a variable from NOAA ERDDAP dataset with structured fallback logic.
    Returns a dict: {"value": float, "source": str}

    Fallback order:
      1. Exact (time, lat, lon)
      2. Temporal fallback (up to `max_days_back` days)
      3. Spatial fallback (nearest valid grid coordinates)
      4. Combined temporal + spatial fallback
      5. spatial + temporal nearest neighbour to input
      6. Raise ValueError if no valid data found
    """

    def _try_request(time_s, la, lo):
        url = (
            f"https://coastwatch.noaa.gov/erddap/griddap/{dataset_id}.csv?"
            f"{variable}[({time_s})][({la})][({lo})]"
        )
        r = requests.get(url, timeout=60)
        df = pd.read_csv(StringIO(r.text), skiprows=[1])

        if df.columns[0].startswith("Error"):
            return np.nan

        if variable not in df.columns:
            return np.nan

        val = df.iloc[0][variable]
        return float(val) if not pd.isna(val) else np.nan

    # Initialise sentinels before the fallback chain
    gc_distance_km = float('inf')
    nearest_lat = lat
    nearest_lon = lon

    # 1) Exact
    val = _try_request(time_str, lat, lon)
    if not np.isnan(val):
        return {"value": val, "source": "exact"}

    # 2) Temporal fallback
    t0 = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
    for d in range(1, max_days_back + 1):
        t_alt = (t0 - timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
        val = _try_request(t_alt, lat, lon)
        if not np.isnan(val):
            return {"value": val, "source": f"temporal_-{d}d"}

    # 3) Spatial fallback: nearest grid point WITH distance check
    tree, grid_points = _get_erddap_grid(dataset_id)
    dist, idx = tree.query([lat, lon]) #tree.query returns 2 parameters from cKDtree object; dist = (euclidean?) distance of nearest (lat, lon) pair to input input (lat, lon) pair - irrelevant for this code; idx = index position of (lat, lon) pair nearest to input (lat, lon) pair in cKDTree
    nearest_lat, nearest_lon = grid_points[idx]

    # Convert Euclidean dist to great-circle distance (km)
    gc_distance_km = haversine(lat, lon, nearest_lat, nearest_lon) / 1000

    # Only accept if within reasonable distance for coral reefs
    if gc_distance_km <= max_km:
        val = _try_request(time_str, nearest_lat, nearest_lon)
        if not np.isnan(val):
            return {
                "value": val,
                "source": f"spatial_nearest({nearest_lat:.3f},{nearest_lon:.3f},{gc_distance_km:.1f}km)"
            }

        # 4) Combined fallback on SAME nearest point (still within max_km)
        for d in range(1, max_days_back + 1):
            t_alt = (t0 - timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
            val = _try_request(t_alt, nearest_lat, nearest_lon)
            if not np.isnan(val):
                return {
                    "value": val,
                    "source": f"combined(-{d}d,{gc_distance_km:.1f}km_nearest)",
                    "used_coordinates": f"spatial_nearest({nearest_lat:.3f},{nearest_lon:.3f})"
                }

    # 5) FINAL: True 4D nearest neighbor (wider bounds)
    nearest_val, nearest_source = _try_nearest_4d(dataset_id, variable, time_str, lat, lon)
    if not np.isnan(nearest_val):
        return {"value": nearest_val, "source": nearest_source}

    # 6) No data within acceptable range
    raise ValueError(
        f"No valid '{variable}' data within {max_km}km of ({lat:.3f}, {lon:.3f}) "
        f"around {time_str}. Nearest grid at {gc_distance_km:.1f}km away @ {nearest_lat}, {nearest_lon}."
    )

def compute_weekly_clim_max_parallel(lat, lon, dt, years_back=10):
    end_year = dt.year
    start_year = max(end_year - years_back, 1981)
    years = range(start_year, end_year)

    def fetch_year(year):
        t_start = f"{year}-01-01"
        t_end   = f"{year}-12-31"
        url = (
            f"https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily.csv?"
            f"analysed_sst"
            f"[({t_start}):({t_end})][({lat})][({lon})]"
        )
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                return None
            df = pd.read_csv(StringIO(r.text), skiprows=[1])
            if df.empty or "analysed_sst" not in df.columns:
                return None
            df["analysed_sst"] = pd.to_numeric(df["analysed_sst"], errors="coerce")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            return df.dropna(subset=["analysed_sst", "time"])
        except Exception:
            return None



    all_dfs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_year, y): y for y in years}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_dfs.append(result)

    if not all_dfs:
        raise ValueError(f"No SST data returned for ({lat}, {lon}) within a {years_back} year window from {dt}, therefore could not compute TSA/TSA_DHW.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["week"] = combined["time"].dt.isocalendar().week.astype(int)

    clim_max = (
        combined.groupby("week")["analysed_sst"]
        .max()
        .reindex(range(1, 53))
        .to_dict()
    )

    return clim_max


import time

def fetch_sst_range(lat, lon, end_dt, weeks=12, max_retries=3):
    """
    Fetch SST time series for the last `weeks` weeks ending at end_dt.
    Returns a list of floats (earliest → latest).
    Retries on empty result to handle post-parallel-request server recovery.
    """
    t_end   = end_dt.strftime("%Y-%m-%d")
    t_start = (end_dt - timedelta(weeks=weeks)).strftime("%Y-%m-%d")

    url = (
        f"https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily.csv?"
        f"analysed_sst"
        f"[({t_start}):7:({t_end})][({lat})][({lon})]"
    )

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"[fetch_sst_range] HTTP 429 — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"[fetch_sst_range] HTTP {r.status_code} — retrying in 10s (attempt {attempt+1}/{max_retries})")
                time.sleep(10)
                continue
            df = pd.read_csv(StringIO(r.text), skiprows=[1])
            if df.empty or "analysed_sst" not in df.columns:
                time.sleep(5)
                continue
            df["analysed_sst"] = pd.to_numeric(df["analysed_sst"], errors="coerce")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["analysed_sst", "time"])
            df = df.sort_values("time")
            return df["analysed_sst"].tolist()
        except Exception as e:
            print(f"[fetch_sst_range] Exception: {e} — retrying in 10s (attempt {attempt+1}/{max_retries})")
            time.sleep(10)

    return []

def fetch_environmental_variables(lat, lon, dt):
    """
    Fetch ERDDAP SST/SSTA/SSTA_DHW, compute ClimSST, TSA and TSA_DHW dynamically.
    """

    t_str = dt.strftime("%Y-%m-%dT00:00:00Z")

    # Fetch real ERDDAP variables
    # use e.g. sstDict['source'] to retrieve fallback level for frontend output
    sstDict = erddap_extract("noaacrwsstDaily", "analysed_sst", t_str, lat, lon)
    sst = sstDict["value"]
    sst_source = sstDict["source"]
    sstaDict = erddap_extract("noaacrwsstanomalyDaily", "sea_surface_temperature_anomaly", t_str, lat, lon)
    ssta = sstaDict["value"]
    ssta_source = sstaDict["source"]
    ssta_dhwDict = erddap_extract("noaacrwdhwDaily", "degree_heating_week", t_str, lat, lon)
    ssta_dhw = ssta_dhwDict["value"]
    ssta_dhw_source = ssta_dhwDict["source"]

    clim_sst = sst - ssta  # SST climatology for this day


    # Compute ClimMAX dictionary
    clim_max_dict = compute_weekly_clim_max_parallel(lat, lon, dt, years_back=10)

    # Fetch last 12 weeks SST for TSA computation
    sst_12w = fetch_sst_range(lat, lon, dt, weeks=12)

    if not sst_12w:
        raise ValueError(
            f"fetch_sst_range returned no SST values for ({lat}, {lon}) around {dt}. "
            f"Cannot compute TSA/TSA_DHW."
        )

    # Compute weekly climatology for these weeks
    tsa_values = []
    for i, sst_val in enumerate(sst_12w):
        # Map each week to its ISO week number
        dt_w = dt - timedelta(weeks=12-i)
        week_num = int(dt_w.strftime("%V"))
        clim_max = clim_max_dict.get(week_num, np.nan)
        tsa_values.append(sst_val - clim_max)

    tsa_values = np.array(tsa_values)
    tsa_values[tsa_values < 0] = 0  # only positive anomalies contribute
    tsa = tsa_values[-1]  # latest TSA
    tsa_dhw = np.sum(tsa_values) / 7  # sum over last 12 weeks / 7 = degree heating weeks

    env = {
        "Temperature_Kelvin": float(sst),
        "ClimSST": float(clim_sst),
        "SSTA": float(ssta),
        "SSTA_DHW": float(ssta_dhw),
        "TSA": float(tsa),
        "TSA_DHW": float(tsa_dhw)
    }

    source = {
        "sst_source": sst_source,
        "ssta_source": ssta_source,
        "ssta_dhw_source": ssta_dhw_source

    }

    return env, source


################################################################################

# will not be exact comparted to BCO-DMO (off up to ~200-500m/ 1/2 grid cell?)
# from simplified coastline vectors

# to consider (GSHHG):

# Source	                Effect
# Coastline resolution (i)	±100–250 m
# Small islands omitted	    Underestimate distance
# Projection vs geodesic	±1–2%
# Snapping to polygon edge	±vertex spacing

# reasonable error:

# ±250 m (GSHHG i)
# ±50 m (GSHHG h)

# Vectorized haversine (returns distance in meters)


def distance_to_shore(lat, lon, coast):

    ## creates a geodetic calculator using the WGS84 reference ellipsoid (standard Earth model used by GPS)
    ## more accurate than haversine
    geod = Geod(ellps="WGS84")
    point = Point(lon, lat)

    # Use spatial index to pre-filter candidate geometries
    # critical, this snippet may need to be changed depending on the geopandas version used
    #candidates_idx = list(coast.sindex.nearest(point.bounds, 5, all_matches=False)) # 5 nearest geometries
    candidates_idx = list(coast.sindex.nearest(point, return_all=False))[1].tolist()
    candidates = coast.iloc[candidates_idx]

    min_dist = float('inf')
    for geom in candidates.geometry:
        nearest = nearest_points(point, geom)[1]
        _, _, dist_m = geod.inv(lon, lat, nearest.x, nearest.y)
        if dist_m < min_dist:
            min_dist = dist_m

    return min_dist


################################################################################

# will be off 'reality' due to way of computation vs BCO-DMO
# from gridded bathymetry

#to consider (GEBCO via OpenTopoData):

# Source	                    Typical uncertainty
# Grid resolution (~15″)	    ±225 m horizontally
# Vertical accuracy (shallow)	±1–5 m
# Coastal interpolation	        ±5–20 m


# reasonable error:

# ±10 m (shallow)
# ±20–50 m (deep ocean)


def depth_from_opentopo(lat, lon, timeout=15):
    """
    Returns ocean depth in meters using OpenTopoData (GEBCO 2020).
    Positive = water depth, 0 = land.
    """

    url = (
        "https://api.opentopodata.org/v1/gebco2020"
        f"?locations={lat},{lon}"
    )

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    if not data["results"]:
        return None

    elevation = data["results"][0]["elevation"]

    if elevation is None:
        return None

    # GEBCO convention: negative = ocean
    return float(-elevation) if elevation < 0 else 0.0



################################################################################

## this one is written by AI
## too little scientific knowledge of regions, winds, narrow window cyclone logic and fetch sectors

## these are the rules:

## Exposure is categorical: EXPOSED | SOMETIMES | SHELTERED
## Fetch threshold = 20 km
## Directional (not omnidirectional) (aka corals facing winds)
## Prevailing winds are regional constants
## Cyclone influence is binary (per region)

def infer_region(lat, lon):
    """
    helper function to classify reef region for exposure
    """

    if -40 <= lat <= 30 and -100 <= lon <= -30:
        return "Caribbean"
    if -30 <= lat <= 30 and 30 <= lon <= 150:
        return "IndoPacific"
    if -30 <= lat <= 30 and -150 <= lon <= -70:
        return "E_Pacific"
    return "Other"

def _endpoint(lat, lon, bearing, km):
    """

    """

    ## creates a geodetic calculator using the WGS84 reference ellipsoid (standard Earth model used by GPS)
    geod = Geod(ellps="WGS84")

    lon2, lat2, _ = geod.fwd(lon, lat, bearing, km * 1000)
    return lat2, lon2

def _fetch_direction(lat, lon, bearing, coast):
    """

    """

    ## creates a geodetic calculator using the WGS84 reference ellipsoid (standard Earth model used by GPS)
    geod = Geod(ellps="WGS84")

    MAX_FETCH_KM = 100

    lat2, lon2 = _endpoint(lat, lon, bearing, MAX_FETCH_KM)
    ray = LineString([(lon, lat), (lon2, lat2)])

    bbox = ray.bounds
    candidates = coast.iloc[list(coast.sindex.intersection(bbox))]

    min_dist = MAX_FETCH_KM * 1000

    for geom in candidates.geometry:
        if not ray.intersects(geom):
            continue

        hit = nearest_points(ray, geom)[1]
        _, _, d = geod.inv(lon, lat, hit.x, hit.y)
        min_dist = min(min_dist, d)

    return min_dist / 1000

def _compute_fetch(lat, lon, coast):
    """

    """

    DIRECTIONS = [i * 22.5 for i in range(16)]  # 16-point compass

    return {
        d: _fetch_direction(lat, lon, d, coast)
        for d in DIRECTIONS
    }

def _classify(fetch, region_info):
    """
    actual classifier function for 'Exposure'
    """

    FETCH_THRESHOLD_KM = 20
    NARROW_WINDOW_DEG = 45

    # Rule 1: Facing prevailing winds
    for d in region_info["prevailing_dirs"]:
        nearest = min(fetch, key=lambda b: abs(b - d))
         #print(f"Comparing fetch direction {d}° to nearest {nearest}°: {fetch[nearest]} km")
        if fetch[nearest] >= FETCH_THRESHOLD_KM:
            return "EXPOSED"

    # Rule 2: Narrow window + cyclones — with 360° wrapping fix
    long_dirs = sorted([b for b, f in fetch.items() if f >= FETCH_THRESHOLD_KM])
    #print(f"Long fetch directions: {long_dirs}")

    if long_dirs and region_info["cyclone"]:
        #print(f"Fetch span (°): {max(long_dirs) - min(long_dirs)}")
        # Compute gaps between consecutive directions (including wrap-around gap)
        gaps = [long_dirs[i+1] - long_dirs[i] for i in range(len(long_dirs)-1)]
        gaps.append(360 - long_dirs[-1] + long_dirs[0])  # wrap-around gap
        largest_gap = max(gaps)
        angular_span = 360 - largest_gap  # actual span of fetch directions
        if angular_span <= NARROW_WINDOW_DEG:
            return "SOMETIMES"

    return "SHELTERED"

def classify_exposure(lat, lon, coast):
    """
    Classifies site exposure (BCO-DMO style), pipeline of corresponding functions
    """

    ## regions for 'Exposure'

    REGIONS = {
    "Caribbean": {
        "prevailing_dirs": [45, 60, 75],  # NE trades
        "cyclone": True
        },
    "IndoPacific": {
        "prevailing_dirs": [120, 135, 150],  # SE trades / monsoon
        "cyclone": True
        },
    "E_Pacific": {
        "prevailing_dirs": [270, 300],
        "cyclone": True
        },
    "Other": {
        "prevailing_dirs": [],
        "cyclone": False
        }
    }

    region = infer_region(lat, lon)
    #print(f"Region inferred: {region}")
    region_info = REGIONS[region]
    #print(f"Prevailing wind directions: {REGIONS[region]['prevailing_dirs']}")
    fetch = _compute_fetch(lat, lon, coast)
    #print(f"Fetch for lat={lat}, lon={lon}: {fetch}")

    exposure = _classify(fetch, region_info)

    return exposure


################################################################################

# they likely added a weighting parameter bc numbers are mostly close but slightly off-ish
# NOT a magnitude issue (other evidence, BMO-DCO provides decimal numbers for smth they label occurences within time window..)

def cyclone_frequency(lat, lon):
    """
    Compute cyclone occurences (..) for location across last 50 years.
    BMO-DCO: 1964-2014
    us: 1975-2025
    """

    url = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv")

    storm_ids = set()
    chunks = pd.read_csv(url, usecols=['SID','SEASON','LAT','LON'], skiprows=[1], chunksize=50_000)

    for df in chunks:
        df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        df = df[(df['SEASON'] >= 1975) & (df['SEASON'] <= 2025)]
        df = df[['SID','LAT','LON']].dropna()

        df = df[
            (df["LAT"].between(lat-2, lat+2)) &
            (df["LON"].between(lon-2, lon+2))
        ]

        #for _, row in df.iterrows():
        #    dist = haversine(lat, lon, row['LAT'], row['LON'])
        #    if dist <= 250000: # meters!, not kilometers in our case!
        #        storm_ids.add(row['SID'])

        # Instead of iterrows loop, vectorise
        # Vectorised planar distance approximation (accurate within small bbox):
        df['dist'] = np.sqrt(((df['LAT'] - lat) * 111000)**2 +
                             ((df['LON'] - lon) * 111000 * np.cos(np.radians(lat)))**2)
        df = df[df['dist'] <= 250000]
        storm_ids.update(df['SID'].tolist())

    return len(storm_ids)

###############################################################################

def _get_turbidity_time_bounds():
    """Query ERDDAP metadata to get actual first and last available dates."""
    url = "https://coastwatch.noaa.gov/erddap/info/noaacwNPPVIIRSSQkd490Monthly/index.csv"
    r = requests.get(url, timeout=30)
    df = pd.read_csv(StringIO(r.text))

    time_rows = df[df["Attribute Name"].isin(["time_coverage_start", "time_coverage_end"])]
    bounds = dict(zip(time_rows["Attribute Name"], time_rows["Value"]))

    #t_start = pd.to_datetime(bounds["time_coverage_start"]).strftime("%Y-%m-%d")
    t_end   = pd.to_datetime(bounds["time_coverage_end"]).strftime("%Y-%m-%d")
    #return t_start, t_end
    return t_end

def turbidity(lat, lon):
    """
    Compute mean Kd490 turbidity over the last 10 years from present
    within a 100km buffer (~0.9°) around (lat, lon).
    Uses VIIRS monthly composites via direct ERDDAP HTTP request.
    Matches BCO-DMO: static site property, not date-specific.
    """

    lat_tol = lon_tol = 0.9
    lon = ((lon + 180) % 360) - 180

    lat_min, lat_max = lat - lat_tol, lat + lat_tol
    lon_min, lon_max = lon - lon_tol, lon + lon_tol


    # Last 10 years from last available date
    t_end   = _get_turbidity_time_bounds()
    t_start = (pd.to_datetime(t_end) - pd.DateOffset(years=10)).strftime("%Y-%m-%d")

    print(f"[turbidity] querying ({lat}, {lon}), bbox: lat=[{lat_min},{lat_max}], lon=[{lon_min},{lon_max}]")
    print(f"[turbidity] time window: {t_start} to {t_end}")

    url = (
        f"https://coastwatch.noaa.gov/erddap/griddap/noaacwNPPVIIRSSQkd490Monthly.csv?"
        f"kd_490[({t_start}):1:({t_end})][(0):1:(0)][({lat_min}):5:({lat_max})][({lon_min}):5:({lon_max})]"
    )

    print(f"[turbidity] requesting {t_start} to {t_end}")
    import time
    t0 = time.time()
    r = requests.get(url, timeout=60)
    print(f"[turbidity] HTTP {r.status_code} — download took {time.time()-t0:.1f}s")

    if r.status_code != 200:
        print(f"[turbidity] non-200 response: {r.text[:300]}")
        raise ValueError(f"turbidity fetch failed for ({lat}, {lon}): HTTP {r.status_code}")

    df = pd.read_csv(StringIO(r.text), skiprows=[1])
    if df.empty or "kd_490" not in df.columns:
        raise ValueError(f"kd_490 missing in response for ({lat}, {lon})")

    df["kd_490"] = pd.to_numeric(df["kd_490"], errors="coerce")
    vals = df["kd_490"].dropna().values
    vals = vals[(vals >= 0) & (vals <= 5)]

    if vals.size == 0:
        return np.nan

    kd490_static = float(vals.mean())
    print(f"[turbidity] {vals.size} valid pixels — mean Kd490={kd490_static:.4f}")
    return kd490_static

###############################################################################

def windspeed(lat, lon, dt):
    """
    Fetch windspeed (m/s) from NOAA Blended Winds Daily dataset.
    Returns scalar wind speed computed from u and v components.
    """

    t = dt.strftime("%Y-%m-%dT00:00:00Z")
    base = "https://coastwatch.noaa.gov/erddap/griddap"
    dataset = "noaacwBlendedWindsDaily"

    # Convert to 0-360 longitude convention used by this dataset
    if lon < 0:
        lon += 360

    # Snap to nearest 0.25° grid point without extra HTTP requests
    lat = round(round(lat / 0.25) * 0.25, 2)
    lon = round(round(lon / 0.25) * 0.25, 2)

    zlev = 0

    url = (
        f"{base}/{dataset}.csv?"
        f"u_wind[({t})][{zlev}][({lat})][({lon})],"
        f"v_wind[({t})][{zlev}][({lat})][({lon})]"
    )

    r = requests.get(url, timeout=15)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text), skiprows=[1])

    if df.empty or "u_wind" not in df.columns or "v_wind" not in df.columns:
        raise ValueError(f"windspeed: missing u_wind/v_wind in response for ({lat}, {lon}) at {t}")

    u = float(df.loc[0, "u_wind"])
    v = float(df.loc[0, "v_wind"])

    return float(np.sqrt(u**2 + v**2))

###############################################################################

def build_X_pred(lat, lon, dt):
    """
    X_pred builder function, executes all functions for fetching environmental features for (lat, lon, dt)
    """

    ## Load once from local (GSHHG: global shoreline dataset)
    ## i = intermediate
    ## h = high
    ## f = full
    coast = gpd.read_file(os.path.join(os.path.dirname(__file__), "gshhg-shp-2.3.7", "GSHHS_h_L1.shp"))
    coast = coast.to_crs("EPSG:4326")

    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d")

    print("\nFetching environmental data...")

    fetch_errors = {}
    fallback = {}
    results = {}

    # --- Individual fetches with per-variable error capture ---

    try:
        env, source = fetch_environmental_variables(lat, lon, dt)
        results["env"] = env
        fallback.update({
            "SST": source.get("sst_source"),
            "SSTA": source.get("ssta_source"),
            "SSTA_DHW": source.get("ssta_dhw_source"),
        })
        print("  ✓ SST/SSTA/DHW/TSA variables")
    except Exception as e:
        fetch_errors["env"] = str(e)
        print(f"  ✗ SST/SSTA/DHW/TSA variables: {e}")

    try:
        dist = distance_to_shore(lat, lon, coast)
        results["dist"] = dist
        print("  ✓ Distance to shore")
    except Exception as e:
        fetch_errors["dist"] = str(e)
        print(f"  ✗ Distance to shore: {e}")

    try:
        depth = depth_from_opentopo(lat, lon)
        results["depth"] = depth
        print("  ✓ Depth")
    except Exception as e:
        fetch_errors["depth"] = str(e)
        print(f"  ✗ Depth: {e}")

    try:
        exp = classify_exposure(lat, lon, coast)
        results["exp"] = exp
        print("  ✓ Exposure")
    except Exception as e:
        fetch_errors["exp"] = str(e)
        print(f"  ✗ Exposure: {e}")

    try:
        turb = turbidity(lat, lon)
        results["turb"] = turb
        print("  ✓ Turbidity")
    except Exception as e:
        fetch_errors["turb"] = str(e)
        print(f"  ✗ Turbidity: {e}")

    try:
        cyc = cyclone_frequency(lat, lon)
        results["cyc"] = cyc
        print("  ✓ Cyclone frequency")
    except Exception as e:
        fetch_errors["cyc"] = str(e)
        print(f"  ✗ Cyclone frequency: {e}")

    try:
        wind = windspeed(lat, lon, dt)
        results["wind"] = wind
        print("  ✓ Windspeed")
    except Exception as e:
        fetch_errors["wind"] = str(e)
        print(f"  ✗ Windspeed: {e}")


    # --- Build prediction dataframe ---
    # Build X_pred — use NaN for any failed fetch so missing_cols check catches it

    env = results.get("env", {}) # bc safe lookup in case "env" doesn't exist, no keyerror

    X_pred = pd.DataFrame([{
        "Latitude_Degrees":   lat,
        "Longitude_Degrees":  lon,
        "Date_Year":          dt.year,
        "Date_Month":         dt.month,
        "Distance_to_Shore":  results.get("dist",  np.nan),
        "Turbidity":          results.get("turb",  np.nan),
        "Cyclone_Frequency":  results.get("cyc",   np.nan),
        "Depth_m":            results.get("depth", np.nan),
        "Exposure":           results.get("exp",   np.nan),
        "ClimSST":            env.get("ClimSST",   np.nan),
        "Temperature_Kelvin": env.get("Temperature_Kelvin",np.nan),
        "Windspeed":          results.get("wind",  np.nan),
        "SSTA":               env.get("SSTA",      np.nan),
        "SSTA_DHW":           env.get("SSTA_DHW",  np.nan),
        "TSA":                env.get("TSA",       np.nan),
        "TSA_DHW":            env.get("TSA_DHW",   np.nan),
    }])


    return X_pred, fetch_errors, fallback


# ---------------- Example ---------------- #

if __name__ == "__main__":
    lat = 23.163
    lon = -82.526
    dt = datetime(2024, 3, 10)
    build_X_pred(lat, lon, dt)
    out = build_X_pred(lat, lon, dt)
    out.to_csv("x_pred_out.csv", index=False)
