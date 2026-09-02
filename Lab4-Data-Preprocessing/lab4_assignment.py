# Lab 4 Assignment - Data Quality Assessment & Preprocessing
# Dataset: Chocolate Sales

import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('/mnt/user-data/uploads/Chocolate_Sales.csv')
print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(f"Shape: {df.shape}")
print(df.head())
print(df.dtypes)

# ============================================================
# TASK 1: Identify Data Quality Issues
# ============================================================
print("\n" + "=" * 60)
print("TASK 1: DATA QUALITY ISSUES")
print("=" * 60)

print("\n1. WRONG DATA TYPE - 'Amount' stored as string (object) with '$' and ','")
print(f"   Example value: '{df['Amount'].iloc[0]}' -> Should be float")

print("\n2. WRONG DATA TYPE - 'Date' stored as string instead of datetime")
print(f"   Example value: '{df['Date'].iloc[0]}'")

print("\n3. MISSING VALUES:")
print(df.isnull().sum())

print("\n4. DUPLICATES:", df.duplicated().sum())

# ============================================================
# FIX DATA TYPES
# ============================================================
df['Amount'] = df['Amount'].str.replace('[$,]', '', regex=True).astype(float)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
print("\n[Fixed] Amount and Date data types corrected.")

# ============================================================
# TASK 2: Apply Missing Value Strategy
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: MISSING VALUE STRATEGY")
print("=" * 60)
print("No missing values found in this dataset.")
print("Strategy chosen (if they existed): MEDIAN IMPUTATION")
print("Reason: Median is robust to outliers, making it better than mean")
print("for skewed distributions like sales data.")

# Demonstrate the strategy
print("\nDemonstration - if Amount had missing values:")
df_demo = df.copy()
df_demo.loc[[10, 50, 100], 'Amount'] = np.nan
print(f"  Missing before: {df_demo['Amount'].isnull().sum()}")
df_demo['Amount'] = df_demo['Amount'].fillna(df_demo['Amount'].median())
print(f"  Missing after:  {df_demo['Amount'].isnull().sum()}")
print(f"  Median used:    {df['Amount'].median():.2f}")

# ============================================================
# TASK 3: Detect and Handle Outliers Using IQR
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: OUTLIER DETECTION & HANDLING (IQR METHOD)")
print("=" * 60)

numerical_cols = ['Amount', 'Boxes Shipped']

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\n{col}:")
    print(f"  Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"  Lower Bound = {lower_bound:.2f}, Upper Bound = {upper_bound:.2f}")
    print(f"  Outliers detected: {len(outliers)}")
    print(f"  Strategy: CAPPING (Winsorization) - replace outliers with bounds")
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"  After capping - Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")

# ============================================================
# TASK 4: Normalize Using Min-Max and Z-Score
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: NORMALIZATION")
print("=" * 60)

for col in numerical_cols:
    # Min-Max Normalization: X' = (X - Xmin) / (Xmax - Xmin)
    col_min = df[col].min()
    col_max = df[col].max()
    df[f'{col}_MinMax'] = (df[col] - col_min) / (col_max - col_min)

    # Z-Score Normalization: Z = (X - mean) / std
    col_mean = df[col].mean()
    col_std = df[col].std()
    df[f'{col}_ZScore'] = (df[col] - col_mean) / col_std

    print(f"\n{col}:")
    print(f"  Min-Max -> Range: [{df[f'{col}_MinMax'].min():.4f}, {df[f'{col}_MinMax'].max():.4f}]")
    print(f"  Z-Score -> Mean: {df[f'{col}_ZScore'].mean():.4f}, Std: {df[f'{col}_ZScore'].std():.4f}")

# ============================================================
# TASK 5: PCA - Apply Only If Features Are Correlated
# ============================================================
print("\n" + "=" * 60)
print("TASK 5: PCA DECISION")
print("=" * 60)

corr = df['Amount'].corr(df['Boxes Shipped'])
print(f"\nCorrelation between Amount and Boxes Shipped: {corr:.4f}")
print(f"\nConclusion: Correlation ≈ {corr:.2f} (near zero = very weak relationship)")
print("PCA is NOT applied because:")
print("  - PCA is beneficial when features are highly correlated (|r| > 0.7)")
print("  - With only 2 numerical features and near-zero correlation,")
print("    PCA would not reduce dimensionality meaningfully")
print("  - Applying PCA here would lose interpretability with no benefit")

# Save
df.to_csv('/mnt/user-data/outputs/Chocolate_Sales_processed.csv', index=False)
print("\n\n✓ Processed dataset saved.")
