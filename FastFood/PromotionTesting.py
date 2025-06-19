# Created by Chad Henry
# Created 6.10.2025 | Last modified 6.16.2025
# Purpose: Prepare and explore promotional sales data for analysis and dashboarding.
# Dataset: Weekly sales across multiple stores with 3 randomized promotions.

# ────────────────────────────────────────────────────────────────
# 1) IMPORT LIBRARIES
# ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf

# ────────────────────────────────────────────────────────────────
# 2) LOAD DATA
# ────────────────────────────────────────────────────────────────

df = pd.read_csv('FastFoodMarketingData.csv')

# ────────────────────────────────────────────────────────────────
# 3) INITIAL DATA EXPLORATION
# ────────────────────────────────────────────────────────────────

def explore(data):
    print("DataFrame info:")
    print(data.info(), "\n")
    print("First 5 rows:")
    print(data.head(), "\n")
    print("Summary statistics:")
    print(data.describe(), "\n")
    print("Missing values (%):")
    print((data.isnull().mean() * 100).round(2), "\n")

explore(df)

# ────────────────────────────────────────────────────────────────
# 4) RANDOMIZATION CHECKS
# ────────────────────────────────────────────────────────────────

print("Distribution of treatments:")
print(df['Promotion'].value_counts())

print("\nTreatment assignment by Market Size:")
print(pd.crosstab(df['MarketSize'], df['Promotion']))

# ────────────────────────────────────────────────────────────────
# 5) BALANCE CHECK: AGE OF STORE
# ────────────────────────────────────────────────────────────────

def smd_numeric(x1, x2):
    """Calculate standardized mean difference between two numeric groups."""
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(), x2.std()
    pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
    return abs(m1 - m2) / pooled_sd

print("\nAverage Age of Store by Promotion:")
print(df.groupby('Promotion')['AgeOfStore'].mean())

print("\nStandardized Mean Differences (SMDs):")
print("Promo 1 vs Promo 2:", smd_numeric(df[df['Promotion'] == 1]['AgeOfStore'], df[df['Promotion'] == 2]['AgeOfStore']))
print("Promo 1 vs Promo 3:", smd_numeric(df[df['Promotion'] == 1]['AgeOfStore'], df[df['Promotion'] == 3]['AgeOfStore']))
print("Promo 2 vs Promo 3:", smd_numeric(df[df['Promotion'] == 2]['AgeOfStore'], df[df['Promotion'] == 3]['AgeOfStore']))

# ────────────────────────────────────────────────────────────────
# 6) TIME TRENDS
# ────────────────────────────────────────────────────────────────

print("\nAverage Sales by Week:")
print(df.groupby('week')['SalesInThousands'].mean())

# ────────────────────────────────────────────────────────────────
# 7) RELATIONSHIP BETWEEN AGE AND SALES
# ────────────────────────────────────────────────────────────────

print("\nPearson correlation between AgeOfStore and SalesInThousands:")
print(stats.pearsonr(df['AgeOfStore'], df['SalesInThousands']))

plt.figure(figsize=(12, 6))
sns.scatterplot(x="AgeOfStore", y="SalesInThousands", data=df)
plt.title("Age of Store vs Sales")
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────
# 8) AGE BALANCE BOXPLOT
# ────────────────────────────────────────────────────────────────

plt.figure(figsize=(8, 5))
sns.boxplot(x='Promotion', y='AgeOfStore', data=df)
plt.title('Age of Store by Promotion Group')
plt.xlabel('Promotion Group')
plt.ylabel('Age of Store (Years)')
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────
# 9) SALES DISTRIBUTION CHECKS
# ────────────────────────────────────────────────────────────────

print("\nAverage Sales by Promotion Group:")
print(df.groupby('Promotion')['SalesInThousands'].mean())

# Boxplot of sales
sns.boxplot(x='Promotion', y='SalesInThousands', data=df)
plt.title("Sales Distribution by Promotion")
plt.show()

# Overlayed Histograms by Promotion
plt.figure(figsize=(8, 5))

promo1 = df[df['Promotion'] == 1]['SalesInThousands']
promo2 = df[df['Promotion'] == 2]['SalesInThousands']
promo3 = df[df['Promotion'] == 3]['SalesInThousands']

sns.histplot(promo1, bins=50, kde=True, stat="density", label='Promo 1', color='blue', alpha=0.4)
sns.histplot(promo2, bins=50, kde=True, stat="density", label='Promo 2', color='green', alpha=0.4)
sns.histplot(promo3, bins=50, kde=True, stat="density", label='Promo 3', color='red', alpha=0.4)

plt.title('Sales by Promotion (Histogram)')
plt.xlabel('Sales in Thousands')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Levene's test for variance equality
print("\nLevene test for equal variances in sales across groups:")
print(stats.levene(promo1, promo2, promo3))

# ────────────────────────────────────────────────────────────────
# 10) OUTLIER DETECTION (IQR METHOD)
# ────────────────────────────────────────────────────────────────

def detect_outliers_iqr(data, group_col, value_col):
    outliers = []
    for group in data[group_col].unique():
        group_data = data[data[group_col] == group][value_col]
        q1 = group_data.quantile(0.25)
        q3 = group_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        group_outliers = data[(data[group_col] == group) & ((group_data < lower) | (group_data > upper))]
        outliers.append(group_outliers)
    return pd.concat(outliers)

outliers_df1 = detect_outliers_iqr(df[df['Promotion'] == 1], 'Promotion', 'SalesInThousands')
outliers_df2 = detect_outliers_iqr(df[df['Promotion'] == 2], 'Promotion', 'SalesInThousands')
outliers_df3 = detect_outliers_iqr(df[df['Promotion'] == 3], 'Promotion', 'SalesInThousands')

print("\nOutliers by Group:")
print(f'Promo 1 outliers:\n{outliers_df1}\n')
print(f'Promo 2 outliers:\n{outliers_df2}\n')
print(f'Promo 3 outliers:\n{outliers_df3}\n')

# ────────────────────────────────────────────────────────────────
# 11) OLS REGRESSION WITH CLUSTERED STANDARD ERRORS
# ────────────────────────────────────────────────────────────────

# Week included as categorical variable to check for time effects
model = smf.ols("SalesInThousands ~ C(Promotion) + AgeOfStore + C(MarketSize) + C(week)", data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['LocationID']}
)
print("\nOLS with Raw Sales:")
print(model.summary())

# ────────────────────────────────────────────────────────────────
# 12) LOG-TRANSFORMED MODEL FOR ROBUSTNESS
# ────────────────────────────────────────────────────────────────

df['log_sales'] = np.log1p(df['SalesInThousands'])

model = smf.ols("log_sales ~ C(Promotion) + AgeOfStore + C(MarketSize)", data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['LocationID']}
)
print("\nOLS with Log-Transformed Sales:")
print(model.summary())


print('\n\nWith outlers removed----------------\n\n')
# Rerun model
model = smf.ols("log_sales ~ C(Promotion) + AgeOfStore + C(MarketSize)", data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['LocationID']}
)
print(model.summary())


# ────────────────────────────────────────────────────────────────
# 12) WITH OUTLIERS REMOVED 
# ────────────────────────────────────────────────────────────────


all_outliers = pd.concat([outliers_df1, outliers_df2, outliers_df3])
df_no_outliers = df.drop(all_outliers.index)


model = smf.ols("SalesInThousands ~ C(Promotion) + AgeOfStore + C(MarketSize) + C(week)", data=df_no_outliers).fit(
    cov_type='cluster', cov_kwds={'groups': df_no_outliers['LocationID']}
)
print(model.summary())



# Rerun model
model = smf.ols("log_sales ~ C(Promotion) + AgeOfStore + C(MarketSize)", data=df_no_outliers).fit(
    cov_type='cluster', cov_kwds={'groups': df_no_outliers['LocationID']}
)
print(model.summary())
