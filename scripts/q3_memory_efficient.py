"""
Q3 – Memory-Efficient Processing of Large Dataset
==================================================
Dataset : data/server_logs.csv
Strategy: chunk-based iteration – never load the full file at once.

Memory Optimisation Notes
--------------------------
1. pd.read_csv(..., chunksize=N)
   → Returns a TextFileReader iterator; only N rows reside in RAM at a time.
   → The rest of the file stays on disk until the next iteration.

2. usecols=[...]
   → Only the four required columns are parsed; all other columns are
     skipped by the CSV parser, saving both parse time and RAM.

3. dtype={'ResponseTime': 'float32', 'UserID': 'category'}
   → float32 uses half the memory of float64 for ResponseTime.
   → 'category' dtype for UserID stores repeated strings as integer codes,
     drastically reducing memory for high-cardinality or repeated strings.

4. Incremental aggregation with a plain dict
   → Instead of collecting all ERROR rows into a growing DataFrame, we
     update running (sum, count) pairs per UserID in a lightweight dict.
   → This keeps peak memory proportional to the number of unique UserIDs,
     not the number of ERROR rows.

5. del chunk / gc.collect() (optional)
   → Explicitly drop each chunk and force a GC pass to free memory sooner,
     useful when processing genuinely huge files (1.5 M+ rows).
"""

import pandas as pd
import gc

# ── Config ────────────────────────────────────────────────────────────
FILE_PATH  = "../data/server_logs.csv"
CHUNK_SIZE = 500          # smaller chunk for demo; use 50_000+ in production
COLS       = ["Timestamp", "UserID", "Action", "ResponseTime"]

# Running aggregation state: { UserID: [total_response_time, count] }
user_stats: dict[str, list] = {}
total_rows   = 0
error_rows   = 0

# ── Chunk-based processing ────────────────────────────────────────────
reader = pd.read_csv(
    FILE_PATH,
    usecols=COLS,                           # load only needed columns
    dtype={"ResponseTime": "float32",       # half-precision float → less RAM
           "UserID": "str"},
    chunksize=CHUNK_SIZE                    # iterator, not full load
)

for chunk in reader:
    total_rows += len(chunk)

    # Filter to ERROR rows only
    errors = chunk[chunk["Action"] == "ERROR"]
    error_rows += len(errors)

    # Accumulate sum + count per UserID (avoids growing a big DataFrame)
    for uid, rt in zip(errors["UserID"], errors["ResponseTime"]):
        if uid not in user_stats:
            user_stats[uid] = [0.0, 0]
        user_stats[uid][0] += float(rt)
        user_stats[uid][1] += 1

    del chunk        # release chunk memory immediately
    gc.collect()     # hint to GC to free unreferenced objects

# ── Compute averages and find top 7 ──────────────────────────────────
avg_rt = {
    uid: total / count
    for uid, (total, count) in user_stats.items()
    if count > 0
}

top7 = sorted(avg_rt.items(), key=lambda x: x[1], reverse=True)[:7]

# ── Print results ─────────────────────────────────────────────────────
print(f"Total rows processed : {total_rows:,}")
print(f"ERROR rows found     : {error_rows:,}")
print(f"Unique UserIDs (ERR) : {len(avg_rt)}\n")

print("═" * 42)
print("  Top 7 Users – Highest Avg Response Time")
print("═" * 42)
print(f"  {'Rank':<5} {'UserID':<10} {'Avg ResponseTime (ms)':>22}")
print("─" * 42)
for rank, (uid, avg) in enumerate(top7, start=1):
    print(f"  {rank:<5} {uid:<10} {avg:>22.2f}")
print("═" * 42)
