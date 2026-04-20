"""
Q1 – Smart Missing Value Handling with Business Rules
======================================================
Dataset : data/retail_sales.csv
Output  : outputs/cleaned_retail_sales.csv
"""

import pandas as pd

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv("../data/retail_sales.csv")
print(f"Total rows          : {len(df)}")
print(f"Missing SalesAmount : {df['SalesAmount'].isna().sum()}\n")

# ── Initialise helper columns ──────────────────────────────────────────
df["Imputation_Method"] = None          # will be filled only for missing rows

# ── Impute missing values ──────────────────────────────────────────────
regional_count = 0
category_count = 0

missing_mask = df["SalesAmount"].isna()

for idx in df[missing_mask].index:
    cat    = df.at[idx, "ProductCategory"]
    region = df.at[idx, "Region"]

    # -- Try Regional Median first ----------------------------------------
    regional_subset = df[
        (df["ProductCategory"] == cat) &
        (df["Region"] == region) &
        (~df["SalesAmount"].isna())          # exclude missing rows themselves
    ]["SalesAmount"]

    if not regional_subset.empty:
        df.at[idx, "SalesAmount"]        = regional_subset.median()
        df.at[idx, "Imputation_Method"]  = "Regional_Median"
        regional_count += 1
    else:
        # -- Fallback: Category-wide Median --------------------------------
        category_subset = df[
            (df["ProductCategory"] == cat) &
            (~df["SalesAmount"].isna())
        ]["SalesAmount"]

        if not category_subset.empty:
            df.at[idx, "SalesAmount"]       = category_subset.median()
            df.at[idx, "Imputation_Method"] = "Category_Median"
            category_count += 1
        else:
            # Edge case: no data at all – leave as NaN and mark
            df.at[idx, "Imputation_Method"] = "Could_Not_Impute"

# ── Save cleaned file ──────────────────────────────────────────────────
df.to_csv("../outputs/cleaned_retail_sales.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────
print("═" * 45)
print("  Imputation Summary")
print("═" * 45)
print(f"  Regional_Median  : {regional_count} rows")
print(f"  Category_Median  : {category_count} rows")
print("═" * 45)
print(f"\nCleaned file saved → outputs/cleaned_retail_sales.csv")
print(f"Remaining NaN rows : {df['SalesAmount'].isna().sum()}")
