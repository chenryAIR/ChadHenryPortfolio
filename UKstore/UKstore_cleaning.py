# Created by Chad Henry
# Created 6.8.2025
# Last modified 6.12.2025
# Purpose: Prepare the data for the Airbnb analysis and dashboard.

import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────
# 1) LOAD & MERGE DATA
# ────────────────────────────────────────────────────────────────
print("Loading data...")
df1 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df2 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
print("Data loaded\n")

# APPEND THE TWO FILES
df1['datayear'] = "2010-2011"
df2['datayear'] = "2009-2010"
df = pd.concat([df1,df2], axis = 0)

print(f'Combined df length: {len(df)}\n')


# ────────────────────────────────────────────────────────────────
# 2) DATA EXPLORATION & STANDARDIZATION
# ────────────────────────────────────────────────────────────────
def explore(data):
    print("DataFrame info:")
    print(data.info(), "\n")

    print("First 5 rows:")
    print(data.head(), "\n")

    print("Numeric summary:")
    print(data.describe(), "\n")

    print("Percent missing per column:")
    print((data.isnull().mean() * 100).round(2), "\n")

explore(df)

# Strip whitespace and normalize casing
df['StockCode']   = df['StockCode'].astype(str).str.strip().str.upper()
df['Description'] = df['Description'].astype(str).str.strip().str.lower()
df['Country']     = df['Country'].astype(str).str.strip()

# ────────────────────────────────────────────────────────────────
# 3) EXPLORE & REMOVE INVALID TRANSACTIONS
# ────────────────────────────────────────────────────────────────

def dropfun(df, drop):
    x = len(df)
    df = eval(drop)
    print(f'{x - len(df)} rows were dropped.')
    return df

# ────────────────────────────────────────────────────────────────
#   3a) 0 values in Price and Quantity
# ────────────────────────────────────────────────────────────────

#   Check for 0s and explore
qz = df['Quantity'].value_counts()
if 0 in qz:
    print(f'Price has {qz[0]} 0s values\n')
    print(f'NOTE: Original data did not have this issue so not explored.\n')

pz = df['Price'].value_counts()
if 0 in pz:
    print(f'Price has {pz[0]} 0 values \n')
    
    #   0's related to missing data in the customer id and description?
    zCheck = df.loc[df['Price'] == 0]['Customer ID'].isna().value_counts()
    print(f"{zCheck.get(True,0)} Customer IDs are missing when price == 0. {zCheck.get(False,0)} are not missing\n")
    print('Transactions where price == 0 and costumer ID is not missing:\n')
    print(df.loc[(df["Price"] == 0) & (~df['Customer ID'].isna())], '\n')
    
    #   NOTE: Most are missing, feel comfortable dropping these
    print("Dropping transactions where price == 0.")
    df = dropfun(df, "df.loc[df['Price'] != 0].copy()")
    

# ────────────────────────────────────────────────────────────────
#   3b) Stock codes that are different seem to indicate non-transactions
# ────────────────────────────────────────────────────────────────
# The ones that start with a letter are the most likely not a transaction
print(df.loc[df['StockCode'].astype(str).str[0].str.isalpha()]['Description'].value_counts(),'\n')
#   NOTE: Manually checking the output to create a list of desciptions that
#           are not transactions or ones we won't include

non_products = ['postage','dotcom postage','amazon fee','bank charges',
        'manual','discount','cruk commission','this is a test product.']
#    NOTE: Multiple versions have 'adjustement' and we'll handle those differently

mask_nonprod = (df['Description'].isin(non_products) | df['Description'].str.contains('adjust', na=False))
print('Dropping non-sales/return transactions:\n')
df = dropfun(df,"df.loc[~mask_nonprod].copy()")

# ────────────────────────────────────────────────────────────────
#   3b) Duplicates
# ────────────────────────────────────────────────────────────────
dup = df.duplicated().value_counts()[True]
print(f"There are {dup} ({round(dup/len(df)*100,1)}%) duplicate records")
#NOTE: Since there are so few, I'm assuming they are errors and not legit transaction
print("Dropping duplicate transactions:\n")
df = dropfun(df, "df.drop_duplicates(keep = 'first').copy()")

# ────────────────────────────────────────────────────────────────
# 4) RESOLVE STOCKCODE AND DESCRIPTION MISMATCHES
# ────────────────────────────────────────────────────────────────

# Get unique pairs and find codes mapping to >1 description
pairs = df.loc[:,['StockCode','Description']].drop_duplicates()
desc_counts = pairs.groupby('StockCode')['Description'].nunique()
bad_codes = desc_counts[desc_counts > 1].index

if len(bad_codes):
    print("Found StockCodes with multiple Descriptions:")
    for code in bad_codes:
        variants = pairs.loc[pairs['StockCode']==code, 'Description'].tolist()
        print(f"  {code}: {variants}")
    # Resolve by using the most common description for each code
    mode_desc = (df.groupby('StockCode')['Description'].agg(lambda s: s.value_counts().idxmax()))
    df['Description'] = df['StockCode'].map(mode_desc)
    print("Fixed mismatched descriptions by choosing the most frequent.\n")


# ────────────────────────────────────────────────────────────────
# 4) CALCULATE NEW VARIABLES
# ────────────────────────────────────────────────────────────────
df['Sales'] = df['Quantity'] * df['Price']

df['IsReturn'] = np.where(df['Invoice'].astype(str).str.startswith('C') | (df['Quantity'] < 0), 1, 0)
print("Return flag counts:\n", df['IsReturn'].value_counts(), "\n")

first_dates = df.groupby('Customer ID')['InvoiceDate'].min()
df['FirstPurchaseDate'] = df['Customer ID'].map(first_dates)
df['IsFirstPurchase'] = np.where(df['InvoiceDate'] == df['FirstPurchaseDate'], 1, 0)
print("First-purchase flag counts:\n", df['IsFirstPurchase'].value_counts(), "\n")

explore(df[['IsReturn', 'IsFirstPurchase']])
print ("Sales by return status")
print('Returns')
explore(df.loc[df['IsReturn'] == 1, ['Sales']])
print('Sales')
explore(df.loc[df['IsReturn'] == 0, ['Sales']])
print ("Quatitiy by return status")
print('Returns')
explore(df.loc[df['IsReturn'] == 1, ['Quantity']])
print('Sales')
explore(df.loc[df['IsReturn'] == 0, ['Quantity']])

#Date parts
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['MonthName'] = df['InvoiceDate'].dt.strftime('%B')
df['Weekday'] = df['InvoiceDate'].dt.day_name()


# ────────────────────────────────────────────────────────────────
# 5) FINAL CHECKS
# ────────────────────────────────────────────────────────────────

## Let's explore the data with crosstabs and the important data
pd.crosstab(index=df['Country'], columns=df['IsReturn'], normalize='index').sort_values(1, ascending= False)
pd.crosstab(index=df['datayear'], columns=df['IsReturn'], normalize='index').sort_values(1, ascending= False)
pd.crosstab(index=df['Country'], columns=df['IsFirstPurchase'], normalize='index').sort_values(1, ascending= False)
pd.crosstab(index=df['datayear'], columns=df['IsFirstPurchase'], normalize='index').sort_values(1, ascending= False)
pd.crosstab(index=df['Country'], columns=df['datayear'], values=df['Sales'], aggfunc='sum')
    #NOTE:Nigeria only has negative values. 

#Overall chech again
explore(df)

# === Step 11: Export cleaned data ===
output_file = "cleaned_online_retail_data.csv"
df.to_csv(output_file, index=False)
print(f"Cleaned data exported to: {output_file}")


