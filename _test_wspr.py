import urllib.request, urllib.parse, json, sys

BASE = "https://db1.wspr.live/"

def run(q):
    url = BASE + "?" + urllib.parse.urlencode({"query": q})
    req = urllib.request.Request(url, headers={"User-Agent": "FadeWatch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")

# Aggregate to one row per transmission: (tx_sign, time) → best SNR heard, tx location
# This is what actually feeds into per-grid time series for outage prediction
q = ("SELECT tx_sign, tx_loc, tx_lat, tx_lon, time, band, frequency, "
     "max(snr) as snr, count() as rx_count "
     "FROM wspr.rx "
     "WHERE time >= '2024-01-01 00:00:00' AND time < '2024-02-01 00:00:00' "
     "AND band IN (1,3,5,7) "
     "AND tx_sign IN ('WB6CXC','G4HSB','F4WCV','ON5SE','KE7GZ') "
     "GROUP BY tx_sign, tx_loc, tx_lat, tx_lon, time, band, frequency "
     "ORDER BY tx_sign, time FORMAT JSONCompact")
data = json.loads(run(q))
rows = data["data"]
sys.stdout.buffer.write(f"Transmissions (aggregated) for top-5 tx, Jan 2024: {len(rows):,}\n".encode())
sys.stdout.buffer.write(f"Sample row: {rows[0]}\n".encode())

# Now project: top 200 tx_sign, aggregated to 1 row per transmission
q2 = ("SELECT tx_sign, count() as n_transmissions "
      "FROM (SELECT tx_sign, time, band FROM wspr.rx "
      "      WHERE time >= '2024-01-01 00:00:00' AND time < '2024-02-01 00:00:00' "
      "      AND band IN (1,3,5,7) "
      "      GROUP BY tx_sign, time, band) "
      "GROUP BY tx_sign ORDER BY n_transmissions DESC LIMIT 200 FORMAT JSONCompact")
data2 = json.loads(run(q2))
rows2 = data2["data"]
total_tx = sum(int(r[1]) for r in rows2)
top5_tx = [(r[0], int(r[1])) for r in rows2[:5]]
sys.stdout.buffer.write(f"\nTop 200 tx_sign, aggregated transmissions: {total_tx:,}/month\n".encode())
sys.stdout.buffer.write(f"~{total_tx*130/1e6:.0f} MB/month uncompressed | ~{total_tx*22/1e6:.0f} MB/month gzipped\n".encode())
sys.stdout.buffer.write(f"~{total_tx*22*32/1e9:.1f} GB total (32 months)\n".encode())
sys.stdout.buffer.write(f"Top 5: {top5_tx}\n".encode())
