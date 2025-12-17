from functools import lru_cache
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from datetime import datetime, timedelta
#from math import radians, sin, cos, asin, sqrt, atan2
from functools import lru_cache
from io import StringIO
import math
import xarray as xr
from pyproj import Geod
from pathlib import Path


@lru_cache(maxsize=256)
def erddap_extract(dataset_id, variable, time_str, lat, lon):
    """
    Extractor for ERDDAP server temperature anomaly parameters.
    """

    url = (
        f"https://coastwatch.noaa.gov/erddap/griddap/{dataset_id}.csv?"
        f"{variable}[({time_str})][({lat})][({lon})]"
    )

    r = requests.get(url, timeout=15)
    df = pd.read_csv(StringIO(r.text))

    if df.columns[0].startswith("Error"):
        raise RuntimeError(f"ERDDAP error:\n{r.text}")

    if variable not in df.columns:
        raise KeyError(
            f"Variable '{variable}' not found in ERDDAP response. "
            f"Available columns: {list(df.columns)}"
        )

    return float(df.iloc[1][variable])


def fetch_sst_range(lat, lon, end_dt, weeks=12):
    """
    Fetch SST time series for the last weeks ending at end_dt.
    Returns a list of floats.
    """

    sst_values = []
    for w in range(weeks):
        dt = end_dt - timedelta(weeks=w)
        time_str = dt.strftime("%Y-%m-%dT00:00:00Z")
        try:
            sst = erddap_extract("noaacrwsstDaily", "analysed_sst", time_str, lat, lon)
            sst_values.append(sst)
        except RuntimeError:
            continue  # skip missing days
    return sst_values[::-1]  # earliest → latest


def compute_weekly_clim_max(lat, lon, dt, years_back=10):
    """
    Compute weekly climatological maximum SST for the same week of the year,
    based on the past `years_back` years. Returns dict {week_num: max_sst}.
    """

    clim_max = {}
    end_year = dt.year
    start_year = max(end_year - years_back, 1981)

    for week in range(1, 53):
        max_vals = []
        for y in range(start_year, end_year):
            try:
                dt_week = datetime.strptime(f"{y}-W{week}-1", "%G-W%V-%u") #Monday
                time_str = dt_week.strftime("%Y-%m-%dT00:00:00Z")
                sst = erddap_extract("noaacrwsstDaily", "analysed_sst", time_str, lat, lon)
                max_vals.append(sst)
            except:
                continue
        clim_max[week] = np.max(max_vals) if max_vals else np.nan
    return clim_max


def fetch_environmental_variables(lat, lon, dt):
    """
    Fetch ERDDAP SST/SSTA/SSTA_DHW, compute ClimSST, TSA and TSA_DHW dynamically.
    """

    t_str = dt.strftime("%Y-%m-%dT00:00:00Z")

    # Fetch real ERDDAP variables
    sst = erddap_extract("noaacrwsstDaily", "analysed_sst", t_str, lat, lon)
    ssta = erddap_extract("noaacrwsstanomalyDaily", "sea_surface_temperature_anomaly", t_str, lat, lon)
    ssta_dhw = erddap_extract("noaacrwdhwDaily", "degree_heating_week", t_str, lat, lon)

    clim_sst = sst - ssta  # SST climatology for this day


    # Compute ClimMAX dictionary
    clim_max_dict = compute_weekly_clim_max(lat, lon, dt, years_back=10)

    # Fetch last 12 weeks SST for TSA computation
    sst_12w = fetch_sst_range(lat, lon, dt, weeks=12)

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


    return {
        "ClimSST": clim_sst,
        "SSTA": ssta,
        "SSTA_DHW": ssta_dhw,
        "TSA": tsa,
        "TSA_DHW": tsa_dhw
    }

################################################################################


@lru_cache(maxsize=2048)
def fetch_air_temperature_k(lat, lon, dt):
    """
    Get air temperature close to sea surface from NASA Earth observations
    """

    date_str = dt.strftime("%Y%m%d")

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M"
        f"&start={date_str}&end={date_str}"
        f"&latitude={lat}&longitude={lon}"
        f"&community=AG"
        f"&format=JSON"
    )

    r = requests.get(url, timeout=20)
    data = r.json()

    if "properties" not in data:
        raise ValueError(f"NASA POWER error: {data}")

    params = data["properties"].get("parameter", {})
    if "T2M" not in params or date_str not in params["T2M"]:
        raise ValueError(f"T2M missing for {date_str} at ({lat},{lon})")

    return params["T2M"][date_str] + 273.15


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


def distance_to_shore(lat, lon, coast):
    """
    Returns distance to nearest coastline in meters (accurate to polygon edges).
    """

    ## creates a geodetic calculator using the WGS84 reference ellipsoid (standard Earth model used by GPS)
    geod = Geod(ellps="WGS84")

    point = Point(lon, lat)
    min_dist = float('inf')

    for geom in coast.geometry:
        # find the true nearest point on the geometry
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


def depth_from_opentopo(lat, lon, timeout=10):
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

    # Rule 2: Narrow window + cyclones
    long_dirs = [b for b, f in fetch.items() if f >= FETCH_THRESHOLD_KM]
    #print(f"Long fetch directions: {long_dirs}")

    if long_dirs and region_info["cyclone"]:
        #print(f"Fetch span (°): {max(long_dirs) - min(long_dirs)}")
        if max(long_dirs) - min(long_dirs) <= NARROW_WINDOW_DEG:
            return "SOMETIMES"

    # Rule 3
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
        #df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        #df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        df = df[(df['SEASON'] >= 1975) & (df['SEASON'] <= 2025)]
        df = df[['SID','LAT','LON']].dropna()

        df = df[
            (df["LAT"].between(lat-2, lat+2)) &
            (df["LON"].between(lon-2, lon+2))
        ]

        for _, row in df.iterrows():
            dist = haversine(lat, lon, row['LAT'], row['LON'])
            if dist <= 200000: # meters!, not kilometers in our case!
                storm_ids.add(row['SID'])

    return len(storm_ids)

###############################################################################

def turbidity(lat, lon, dt):

    # important: dates back to 2012 only!
    url = "https://coastwatch.noaa.gov/erddap/griddap/noaacwNPPVIIRSSQkd490Daily"
    ds = xr.open_dataset(url, engine="pydap")

    # Fixed bounding box half-width
    # kd490 in 100km window around coordinates
    # 100km ~ 0.9°
    # so query bounding boxes need to be:
    # lat ± 0.9
    # lon ± 0.9
    lat_tol = lon_tol = 0.9

    # Normalize longitude to [-180, 180]
    if lon > 180:
        lon -= 360
    elif lon < -180:
        lon += 360

    # Create bounding box
    lat_min, lat_max = lat - lat_tol, lat + lat_tol
    lon_min, lon_max = lon - lon_tol, lon + lon_tol

     # Latitude (descending) and Longitude (ascending) arrays
    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values

    # Latitude indices (descending)
    # matching coordinates with actual nearest values in dataset
    lat_start_idx = np.searchsorted(lat_vals[::-1], lat_min, side='left')
    lat_stop_idx  = np.searchsorted(lat_vals[::-1], lat_max, side='right')
    lat_slice = slice(len(lat_vals) - lat_stop_idx, len(lat_vals) - lat_start_idx)

    # Longitude indices (ascending)
    # matching coordinates with actual nearest values in dataset
    lon_start_idx = np.searchsorted(lon_vals, lon_min, side='left')
    lon_stop_idx  = np.searchsorted(lon_vals, lon_max, side='right')
    lon_slice = slice(lon_start_idx, lon_stop_idx)

    # Select spatial subset
    sub = ds.isel(latitude=lat_slice, longitude=lon_slice)

    # Select nearest time
    try:
        sub = sub.sel(time=np.datetime64(dt), method="nearest")
    except KeyError:
        print(f"No data for requested date {dt}")
        return np.array([])

    # Extract kd_490 values
    if "kd_490" not in sub:
        print("kd_490 variable not found in subset")
        return np.array([])

    vals = sub["kd_490"].values

    # Filter invalid data
    if vals.size > 0:
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= 0) & (vals <= 5)]

    return float(vals.mean())

###############################################################################

def windspeed(lat, lon, dt):
    """
    Fetch and compute windspeed parametersfor coordinates in .25 degree stepping
    """

    t = dt.strftime("%Y-%m-%dT00:00:00Z")
    base = "https://coastwatch.noaa.gov/erddap/griddap"
    dataset = "noaacwBlendedWindsDaily"

    if lon < 0:
        lon += 360

    # fetch available lat/lon points
    # they have 0.25 spacing
    url_lats = f"{base}/{dataset}.csv?latitude"
    df_lats = pd.read_csv(StringIO(requests.get(url_lats).text), skiprows=[1])
    lats_available = df_lats['latitude'].to_numpy()

    url_lons = f"{base}/{dataset}.csv?longitude"
    df_lons = pd.read_csv(StringIO(requests.get(url_lons).text), skiprows=[1])
    lons_available = df_lons['longitude'].to_numpy()

    # snap to nearest available grid point
    lat = lats_available[np.abs(lats_available - lat).argmin()]
    lon = lons_available[np.abs(lons_available - lon).argmin()]

    # fetch available times
    # not necessary because 'P1D' = 1 entry per day
    #url_times = f"{base}/{dataset}.csv?time"
    #df_times = pd.read_csv(StringIO(requests.get(url_times).text), skiprows=[1])
    #times = pd.to_datetime(df_times['time'], utc=True)

    # snap to nearest available time
    #target = pd.Timestamp(dt).tz_localize('UTC')
    #nearest_time = times.iloc[(times - target).abs().argmin()]
    #t = nearest_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # equals vertical level to sea level
    zlev = 0

    url = (
        f"{base}/{dataset}.csv?"
        f"u_wind[({t})][{zlev}][({lat})][({lon})],"
        f"v_wind[({t})][{zlev}][({lat})][({lon})]"
    )

    r = requests.get(url)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text), skiprows=[1])

    # Column names are exactly these:
    u = float(df.loc[0, "u_wind"])
    v = float(df.loc[0, "v_wind"])

    return float(np.sqrt(u*u + v*v))

###############################################################################

def build_X_pred(lat, lon, dt):
    """
    X_pred builder function, executes all functions for fetching environmental features for (lat, lon, dt)
    """

    ## Load once from local (GSHHG: global shoreline dataset)
    ## i = intermediate
    ## h = high
    ## f = full

    BASE_DIR = Path(__file__).resolve().parent
    coast_path = BASE_DIR / "gshhg-shp-2.3.7" / "GSHHS_h_L1.shp"

    coast = gpd.read_file(coast_path)
    coast = coast.to_crs("EPSG:4326")


    env = fetch_environmental_variables(lat, lon, dt)
    #print(env)
    air_k = fetch_air_temperature_k(lat, lon, dt)
    #print(air_k)
    dist = distance_to_shore(lat, lon, coast)
    #print(dist)
    depth = depth_from_opentopo(lat, lon)
    #print(depth)
    exp = classify_exposure(lat, lon, coast)
    #print(exp)
    turb = turbidity(lat, lon, dt)
    #print(turb)
    cyc = cyclone_frequency(lat, lon)
    #print(cyc)
    wind = windspeed(lat, lon, dt)
    #print(wind)

    return pd.DataFrame(dict(
        Latitude_Degrees=[lat],
        Longitude_Degrees=[lon],
        Date_Year=[dt.year],
        Date_Month=[dt.month],
        Distance_to_Shore=[dist],
        Turbidity=[turb],
        Cyclone_Frequency=[cyc],
        Depth_m=[depth],
        Exposure=[exp],
        ClimSST=[env["ClimSST"]],
        Temperature_Kelvin=[air_k],
        Windspeed=[wind],
        SSTA=[env["SSTA"]],
        SSTA_DHW=[env["SSTA_DHW"]],
        TSA=[env["TSA"]],
        TSA_DHW=[env["TSA_DHW"]]
    ))


# ---------------- Example ---------------- #

if __name__ == "__main__":
    lat = 23.163
    lon = -82.526
    dt = datetime(2024, 3, 10)
    build_X_pred(lat, lon, dt)
    out = build_X_pred(lat, lon, dt)
    out.to_csv("x_pred_out.csv", index=False)
