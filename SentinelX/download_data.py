"""
Download a small set of genuine CIC-IDS2017 CSV source files.
The source files are real labelled network-flow records. They are not
included in this repository because the originals are large.
"""
from pathlib import Path
import requests
from src.config import RAW_DIR

REPO = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"
FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if target.exists() and target.stat().st_size > 1_000_000:
            print(f"[skip] {name} already exists")
            continue
        url = f"{REPO}/{name}?download=true"
        print(f"[download] {name}")
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            done = 0
            with target.open("wb") as out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done/total:.0%}", end="")
        print(f"\n[saved] {target} ({target.stat().st_size/1e6:.1f} MB)")
    print("\nDownload complete. Next run: python prepare_data.py")

if __name__ == "__main__":
    main()
