"""Download every finished arena table's raw JSON into the data directory (skips ones already present)."""
import json, re, sys, urllib.request
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (replay-fetcher)"}
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()

ids, page = [], 1
while page <= 40:
    html = get(f"https://www.pokerbots.ac/matches?page={page}").decode("utf-8", "replace")
    found = [m for m in dict.fromkeys(re.findall(r"/matches/(m_[0-9a-z_]+)", html)) if m not in ids]
    if not found: break
    ids += found; page += 1
new = 0
for mid in ids:
    dest = out / f"{mid}.json"
    if dest.exists(): continue
    html = get(f"https://www.pokerbots.ac/matches/{mid}").decode("utf-8", "replace")
    m = re.search(r"https://[a-z0-9]+\.public\.blob\.vercel-storage\.com/matches/" + re.escape(mid) + r"[A-Za-z0-9_-]*\.json", html)
    if not m:
        print("no raw json link for", mid); continue
    data = get(m.group(0))
    json.loads(data)  # must parse
    dest.write_bytes(data); new += 1
print(f"matches listed: {len(ids)}, newly downloaded: {new}, total on disk: {len(list(out.glob('*.json')))}")
