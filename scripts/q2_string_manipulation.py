"""
Q2 – Complex String Manipulation with Duplicate Handling
=========================================================
Dataset : data/employee_master.csv
Output  : (printed DataFrame – first 8 rows + duplicate count summary)
"""

import pandas as pd
import re

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv("../data/employee_master.csv")
print(f"Total employees : {len(df)}\n")

# ── Step 1 : Remove titles ─────────────────────────────────────────────
TITLES = {"Dr.", "Mr.", "Ms.", "Mrs.", "Prof.", "Er."}

def remove_title(name: str) -> str:
    """Strip any leading title token that ends with a period."""
    tokens = name.strip().split()
    if tokens and tokens[0] in TITLES:
        tokens = tokens[1:]
    return " ".join(tokens)

df["Clean_Name"] = df["Employee_Name"].apply(remove_title)

# ── Step 2 : Split into First / Middle_Initial / Last ─────────────────
def parse_name(name: str):
    """
    Returns (first, middle_initial, last).
    Handles 2-token  : First Last          → middle = ''
    Handles 3+-token : First Middle Last   → middle = first letter of token[1]
    Middle initials already like 'M.' are kept as-is (strip the dot).
    """
    parts = name.split()
    if len(parts) == 1:
        return parts[0], "", ""
    elif len(parts) == 2:
        return parts[0], "", parts[1]
    else:
        # tokens[1] could be "M." or a full middle name
        mid_token = parts[1].rstrip(".")      # remove trailing dot
        middle_initial = mid_token[0].upper() if mid_token else ""
        return parts[0], middle_initial, parts[-1]

parsed = df["Clean_Name"].apply(parse_name)
df["First_Name"]      = parsed.apply(lambda t: t[0])
df["Middle_Initial"]  = parsed.apply(lambda t: t[1])
df["Last_Name"]       = parsed.apply(lambda t: t[2])

# ── Step 3 : Generate unique usernames ────────────────────────────────
seen_usernames: dict[str, int] = {}
duplicates_handled = 0

def generate_username(first: str, last: str) -> str:
    global duplicates_handled
    if not first or not last:
        base = (first or last or "user").lower()
    else:
        base = (first[0] + last).lower()
        base = re.sub(r"[^a-z0-9]", "", base)   # remove non-alphanumeric

    if base not in seen_usernames:
        seen_usernames[base] = 1
        return base
    else:
        seen_usernames[base] += 1
        count = seen_usernames[base]
        duplicates_handled += 1
        return f"{base}{count}"

df["Username"] = df.apply(
    lambda row: generate_username(row["First_Name"], row["Last_Name"]),
    axis=1
)

# ── Display ───────────────────────────────────────────────────────────
display_cols = ["Employee_Name", "First_Name", "Middle_Initial", "Last_Name", "Username"]
print("═" * 75)
print("  First 8 rows of processed DataFrame")
print("═" * 75)
print(df[display_cols].head(8).to_string(index=False))
print("═" * 75)
print(f"\n  Total duplicate usernames handled : {duplicates_handled}")
print(f"  Unique usernames generated        : {df['Username'].nunique()}")
