import requests, os

def download_ibtracs():
    url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
    path = os.path.join("project_logic", "ibtracs", "ibtracs.ALL.list.v04r01.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading IBTrACS data...")
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Done!")

if __name__ == "__main__":
    download_ibtracs()
