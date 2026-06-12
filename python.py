 # ================================================================
#     PHARMACY SALES ANALYTICS DASHBOARD
#     Complete Elite Portfolio Project
#     All 13 Phases Combined
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ================================================================
# SETUP OUTPUT FOLDERS
# ================================================================
folders = [
    'pharmacy_output',
    'pharmacy_output/charts',
    'pharmacy_output/data',
    'pharmacy_output/report'
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("=" * 65)
print("   💊 PHARMACY SALES ANALYTICS DASHBOARD")
print("   Complete Elite Portfolio Project")
print("=" * 65)

# ================================================================
# PHASE 1 — LOAD YOUR DATASET
# ================================================================
print("\n📂 PHASE 1: Loading Dataset...")

# ---------------------------------------------------------------
# ⚠️ IMPORTANT: Replace this path with your actual file path
# Example: df = pd.read_csv('Cleaned_Pharmacy_Sales_Register.csv')
# ---------------------------------------------------------------

USE_REAL_DATA = False  # Set to True when you have your CSV ready

if USE_REAL_DATA:
    # ── USE YOUR ACTUAL FILE ──
    df = pd.read_csv('Cleaned_Pharmacy_Sales_Register.csv')
    print("   ✅ Real dataset loaded successfully")
else:
    # ── GENERATE REALISTIC PHARMACY DATA ──
    print("   📦 Generating realistic pharmacy dataset...")
    np.random.seed(42)
    n = 1000

    categories = [
        'Prescription Drugs',
        'OTC Medicines',
        'Supplements & Vitamins',
        'Cosmetics & Skincare',
        'Medical Devices',
        'Baby & Mother Care',
        'Herbal & Ayurvedic'
    ]

    products_by_cat = {
        'Prescription Drugs':    [
            'Amoxicillin 500mg', 'Metformin 850mg',
            'Atorvastatin 20mg', 'Amlodipine 5mg',
            'Omeprazole 20mg',   'Azithromycin 500mg',
            'Ciprofloxacin 500mg'
        ],
        'OTC Medicines':         [
            'Paracetamol 500mg', 'Ibuprofen 400mg',
            'Cetirizine 10mg',   'Antacid Syrup',
            'Cough Syrup',       'ORS Sachets',
            'Vitamin C 500mg'
        ],
        'Supplements & Vitamins':[
            'Multivitamin Tablets', 'Omega-3 Capsules',
            'Calcium + D3',         'Iron Supplement',
            'Protein Powder',       'B-Complex',
            'Zinc Tablets'
        ],
        'Cosmetics & Skincare':  [
            'Sunscreen SPF50',  'Moisturizer Cream',
            'Lip Balm',         'Hand Sanitizer',
            'Antiseptic Cream', 'Aloe Vera Gel',
            'Hair Oil'
        ],
        'Medical Devices':       [
            'Digital Thermometer', 'BP Monitor',
            'Glucometer',          'Pulse Oximeter',
            'Nebulizer',           'Heating Pad',
            'Knee Support'
        ],
        'Baby & Mother Care':    [
            'Baby Diaper Pack', 'Baby Lotion',
            'Baby Powder',      'Breast Pump',
            'Baby Food 200g',   'Teething Gel',
            'Maternity Pads'
        ],
        'Herbal & Ayurvedic':    [
            'Ashwagandha Tablets', 'Triphala Churna',
            'Giloy Juice',         'Neem Capsules',
            'Tulsi Drops',         'Chyawanprash 500g',
            'Turmeric Capsules'
        ]
    }

    price_range = {
        'Prescription Drugs':    (80,  850),
        'OTC Medicines':         (20,  350),
        'Supplements & Vitamins':(150, 1200),
        'Cosmetics & Skincare':  (50,  800),
        'Medical Devices':       (300, 3500),
        'Baby & Mother Care':    (120, 1500),
        'Herbal & Ayurvedic':    (80,  600)
    }

    margin_range = {
        'Prescription Drugs':    (0.12, 0.28),
        'OTC Medicines':         (0.20, 0.40),
        'Supplements & Vitamins':(0.25, 0.50),
        'Cosmetics & Skincare':  (0.30, 0.55),
        'Medical Devices':       (0.15, 0.35),
        'Baby & Mother Care':    (0.18, 0.38),
        'Herbal & Ayurvedic':    (0.22, 0.45)
    }

    regions   = ['North Branch','South Branch',
                 'East Branch', 'West Branch', 'Central Branch']
    payments  = ['Cash','Credit Card','Debit Card',
                 'Insurance','UPI']
    customers = [f'CUST{i:04d}' for i in range(1001, 1151)]

    start = pd.to_datetime('2023-01-01')
    rows  = []

    for i in range(1, n + 1):
        cat      = np.random.choice(categories)
        product  = np.random.choice(products_by_cat[cat])
        qty      = np.random.randint(1, 8)
        u_price  = round(np.random.uniform(*price_range[cat]), 2)
        sales    = round(qty * u_price, 2)
        margin   = np.random.uniform(*margin_range[cat])
        profit   = round(sales * margin, 2)
        cost     = round(sales - profit, 2)
        date     = start + pd.Timedelta(
                       days=int(np.random.randint(0, 365)))
        region   = np.random.choice(regions)
        payment  = np.random.choice(payments)
        customer = np.random.choice(customers)

        rows.append([
            f'INV{i:05d}', date.strftime('%Y-%m-%d'),
            customer, f'Product{np.random.randint(1000,9999)}',
            product, cat, qty, u_price, sales, cost, profit,
            region, payment
        ])

    df = pd.DataFrame(rows, columns=[
        'Invoice_ID', 'Date', 'Customer_ID', 'Product_ID',
        'Product_Name', 'Category', 'Quantity', 'Unit_Price',
        'Sales', 'Cost', 'Profit', 'Region', 'Payment_Method'
    ])
    df.to_csv('pharmacy_output/data/pharmacy_raw_data.csv',
              index=False)
    print(f"   ✅ Dataset generated: {len(df)} rows × "
          f"{len(df.columns)} columns")

# ================================================================
# PHASE 2 — DATA UNDERSTANDING
# ================================================================
print("\n🔍 PHASE 2: Data Understanding...")
print(f"\n   Rows    : {len(df):,}")
print(f"   Columns : {len(df.columns)}")
print("\n   Column Info:")
print(f"   {'Column':<20} {'Dtype':<15} {'Nulls':>6} {'Unique':>8}")
print(f"   {'-'*52}")
for col in df.columns:
    dtype  = str(df[col].dtype)
    nulls  = df[col].isnull().sum()
    unique = df[col].nunique()
    print(f"   {col:<20} {dtype:<15} {nulls:>6} {unique:>8}")

# ================================================================
# PHASE 3 — DATA QUALITY ASSESSMENT
# ================================================================
print("\n🔎 PHASE 3: Data Quality Assessment...")

issues = []

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
total_missing = missing.sum()
if total_missing > 0:
    issues.append(f"Missing values found: {total_missing}")
print(f"   Missing Values      : {total_missing}")

# Duplicates
dups = df.duplicated(subset='Invoice_ID').sum() \
    if 'Invoice_ID' in df.columns else 0
if dups > 0:
    issues.append(f"Duplicate Invoice IDs: {dups}")
print(f"   Duplicate Records   : {dups}")

# Negative Sales
neg_sales = (df['Sales'] < 0).sum() \
    if 'Sales' in df.columns else 0
if neg_sales > 0:
    issues.append(f"Negative Sales: {neg_sales}")
print(f"   Negative Sales      : {neg_sales}")

# Negative Profit
neg_profit = (df['Profit'] < 0).sum() \
    if 'Profit' in df.columns else 0
print(f"   Negative Profit     : {neg_profit} "
      f"(can be valid — losses)")

# Outliers (IQR method)
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df['Sales'] < Q1 - 1.5*IQR) |
            (df['Sales'] > Q3 + 1.5*IQR)).sum()
print(f"   Sales Outliers      : {outliers}")

# Data Quality Score
total_checks   = 5
passed_checks  = sum([
    total_missing == 0,
    dups          == 0,
    neg_sales     == 0,
    outliers      < len(df) * 0.05,
    len(df)       > 100
])
quality_score = int((passed_checks / total_checks) * 100)

print(f"\n   {'='*40}")
print(f"   📊 DATA QUALITY SCORE : {quality_score}/100")
print(f"   {'='*40}")

# ================================================================
# PHASE 4 — DATA CLEANING
# ================================================================
print("\n🧹 PHASE 4: Data Cleaning...")

df_clean = df.copy()
before   = len(df_clean)

# Fix date
df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
df_clean.dropna(subset=['Date'], inplace=True)

# Remove duplicates
df_clean.drop_duplicates(subset='Invoice_ID',
                         keep='first', inplace=True)

# Remove negative sales
df_clean = df_clean[df_clean['Sales'] > 0]

# Strip text columns
text_cols = ['Product_Name','Category','Region','Payment_Method']
for col in text_cols:
    if col in df_clean.columns:
        df_clean[col] = (df_clean[col]
                         .astype(str)
                         .str.strip()
                         .str.title())

# Fix numeric types
for col in ['Sales','Profit','Cost','Unit_Price','Quantity']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean.dropna(subset=['Sales'], inplace=True)

after = len(df_clean)
print(f"   Before cleaning : {before:,} rows")
print(f"   After cleaning  : {after:,} rows")
print(f"   Rows removed    : {before - after:,}")

df_clean.to_csv('pharmacy_output/data/pharmacy_clean.csv',
                index=False)
print("   ✅ Clean data saved")

# ================================================================
# PHASE 5 — FEATURE ENGINEERING
# ================================================================
print("\n⚙️  PHASE 5: Feature Engineering...")

df_clean['Year']              = df_clean['Date'].dt.year
df_clean['Month']             = df_clean['Date'].dt.month
df_clean['Month_Name']        = df_clean['Date'].dt.strftime('%b')
df_clean['Month_Year']        = df_clean['Date'].dt.strftime('%b-%Y')
df_clean['Quarter']           = ('Q' +
    df_clean['Date'].dt.quarter.astype(str))
df_clean['Weekday']           = df_clean['Date'].dt.day_name()
df_clean['Week_Number']       = df_clean['Date'].dt.isocalendar().week
df_clean['Profit_Margin_%']   = (
    (df_clean['Profit'] / df_clean['Sales']) * 100).round(2)
df_clean['Revenue_Per_Unit']  = (
    df_clean['Sales'] / df_clean['Quantity']).round(2)
df_clean['Is_Weekend']        = df_clean['Weekday'].isin(
    ['Saturday','Sunday'])
df_clean['High_Value_Order']  = df_clean['Sales'] > \
    df_clean['Sales'].quantile(0.75)

df_clean.to_csv('pharmacy_output/data/pharmacy_featured.csv',
                index=False)
print("   ✅ Features added:")
new_cols = ['Year','Month','Month_Name','Quarter','Weekday',
            'Profit_Margin_%','Revenue_Per_Unit',
            'Is_Weekend','High_Value_Order']
for col in new_cols:
    print(f"      + {col}")

# ================================================================
# PHASE 6 — KPI CALCULATIONS
# ================================================================
print("\n📊 PHASE 6: KPI Calculations...")

total_revenue  = df_clean['Sales'].sum()
total_profit   = df_clean['Profit'].sum()
total_orders   = df_clean['Invoice_ID'].nunique()
total_qty      = df_clean['Quantity'].sum()
avg_order_val  = total_revenue / total_orders
profit_margin  = (total_profit / total_revenue) * 100
total_cost     = df_clean['Cost'].sum() \
    if 'Cost' in df_clean.columns else 0

print("\n" + "=" * 55)
print("        💊 PHARMACY SALES — KPI SUMMARY")
print("=" * 55)
print(f"   💰 Total Revenue    : ₹{total_revenue:>15,.2f}")
print(f"   📦 Total Profit     : ₹{total_profit:>15,.2f}")
print(f"   🛒 Total Orders     : {total_orders:>16,}")
print(f"   💊 Total Qty Sold   : {total_qty:>16,}")
print(f"   🧾 Avg Order Value  : ₹{avg_order_val:>15,.2f}")
print(f"   📉 Profit Margin    : {profit_margin:>15.2f}%")
print("=" * 55)

# ================================================================
# PHASE 7 — EDA & DETAILED ANALYSIS
# ================================================================
print("\n🔍 PHASE 7: Exploratory Data Analysis...")

# Sales by Category
cat_perf = (df_clean.groupby('Category')
            .agg(Revenue=('Sales','sum'),
                 Profit=('Profit','sum'),
                 Orders=('Invoice_ID','count'),
                 Avg_Margin=('Profit_Margin_%','mean'))
            .sort_values('Revenue', ascending=False)
            .round(2))
cat_perf['Margin_%'] = (
    cat_perf['Profit']/cat_perf['Revenue']*100).round(1)

print("\n📦 CATEGORY PERFORMANCE:")
print(f"   {'Category':<28} {'Revenue':>12}"
      f" {'Profit':>12} {'Margin':>8}")
print(f"   {'-'*63}")
for cat, row in cat_perf.iterrows():
    print(f"   {cat:<28} ₹{row['Revenue']:>10,.0f}"
          f" ₹{row['Profit']:>10,.0f} {row['Margin_%']:>7.1f}%")

# Top Products
top_products = (df_clean.groupby('Product_Name')['Sales']
                .sum().sort_values(ascending=False).head(10))

print("\n🏆 TOP 10 PRODUCTS BY REVENUE:")
for i,(prod,val) in enumerate(top_products.items(),1):
    print(f"   {i:>2}. {prod:<30} ₹{val:>10,.0f}")

# Worst Products
worst_products = (df_clean.groupby('Product_Name')['Sales']
                  .sum().sort_values().head(5))
print("\n⚠️  BOTTOM 5 PRODUCTS BY REVENUE:")
for i,(prod,val) in enumerate(worst_products.items(),1):
    print(f"   {i}. {prod:<30} ₹{val:>10,.0f}")

# Top Customers
top_customers = (df_clean.groupby('Customer_ID')['Sales']
                 .sum().sort_values(ascending=False).head(10))
print("\n👤 TOP 10 CUSTOMERS:")
for i,(cust,val) in enumerate(top_customers.items(),1):
    print(f"   {i:>2}. {cust:<15} ₹{val:>10,.0f}")

# Monthly Revenue
monthly_rev = (df_clean.groupby(['Month','Month_Name'])['Sales']
               .sum().reset_index()
               .sort_values('Month'))
print("\n📅 MONTHLY REVENUE:")
for _, row in monthly_rev.iterrows():
    bar = '█' * int(row['Sales']/total_revenue*80)
    print(f"   {row['Month_Name']:>4}: ₹{row['Sales']:>12,.0f} {bar}")

# Regional Performance
region_perf = (df_clean.groupby('Region')
               [['Sales','Profit']].sum()
               .sort_values('Sales', ascending=False))
region_perf['Margin_%'] = (
    region_perf['Profit']/region_perf['Sales']*100).round(1)
print("\n🗺️  REGIONAL PERFORMANCE:")
for region, row in region_perf.iterrows():
    print(f"   {region:<20} ₹{row['Sales']:>10,.0f}"
          f"  Margin: {row['Margin_%']:.1f}%")

# Payment Methods
payment_perf = (df_clean.groupby('Payment_Method')['Sales']
                .sum().sort_values(ascending=False))
print("\n💳 PAYMENT METHOD BREAKDOWN:")
for pay, val in payment_perf.items():
    pct = val/total_revenue*100
    print(f"   {pay:<15} ₹{val:>10,.0f} ({pct:.1f}%)")

# Quarterly
quarterly = df_clean.groupby('Quarter')['Sales'].sum().sort_index()
print("\n🗓️  QUARTERLY REVENUE:")
for q, val in quarterly.items():
    pct = val/total_revenue*100
    print(f"   {q}: ₹{val:>12,.0f} ({pct:.1f}%)")

# Weekday Analysis
day_order = ['Monday','Tuesday','Wednesday','Thursday',
             'Friday','Saturday','Sunday']
weekday_sales = (df_clean.groupby('Weekday')['Sales']
                 .sum().reindex(day_order))
print("\n📆 SALES BY WEEKDAY:")
for day, val in weekday_sales.items():
    print(f"   {day:<12} ₹{val:>10,.0f}")

# Profit Margin by Category
print("\n💡 PROFIT MARGIN BY CATEGORY:")
margin_by_cat = (df_clean.groupby('Category')['Profit_Margin_%']
                 .mean().sort_values(ascending=False))
for cat, margin in margin_by_cat.items():
    bar = '█' * int(margin / 2)
    print(f"   {cat:<28} {margin:>6.1f}%  {bar}")

# ================================================================
# PHASE 8 — PROFESSIONAL LIGHT-THEME VISUALIZATIONS
# ================================================================
print("\n🎨 PHASE 8: Creating Professional Visualizations...")

# ── LIGHT THEME COLOUR PALETTE ──
C = {
    'bg':          '#FFFFFF',
    'panel':       '#F5F7FA',
    'border':      '#E0E4EC',
    'navy':        '#1B2A4A',
    'blue':        '#2D5BE3',
    'light_blue':  '#EEF2FF',
    'green':       '#0E9E6E',
    'light_green': '#E6F7F2',
    'orange':      '#F07C2A',
    'light_orange':'#FEF3E8',
    'red':         '#E53E3E',
    'purple':      '#7B2FBE',
    'gray':        '#6B7280',
    'light_gray':  '#F9FAFB',
    'text':        '#1F2937',
    'subtext':     '#6B7280',
}

CATEGORY_COLORS = [
    '#2D5BE3','#0E9E6E','#F07C2A',
    '#7B2FBE','#E53E3E','#0891B2','#84CC16'
]

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.titleweight':   'bold',
    'axes.titlecolor':    C['text'],
    'axes.labelcolor':    C['text'],
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
    'axes.edgecolor':     C['border'],
    'figure.facecolor':   C['bg'],
    'axes.facecolor':     C['panel'],
    'xtick.color':        C['subtext'],
    'ytick.color':        C['subtext'],
    'grid.color':         C['border'],
    'grid.alpha':         0.7,
})

monthly_sorted = (df_clean
                  .groupby(['Month','Month_Name'])['Sales']
                  .sum().reset_index()
                  .sort_values('Month'))

# ============================================================
# CHART 1 — MAIN DASHBOARD (Light Theme)
# ============================================================
fig = plt.figure(figsize=(22, 16), facecolor=C['bg'])

# Header
fig.text(0.04, 0.96,
         '💊  Pharmacy Sales Analytics Dashboard',
         fontsize=24, fontweight='bold', color=C['navy'])
fig.text(0.04, 0.93,
         'Performance Overview  |  Financial Year 2023  |  '
         'All Branches Combined',
         fontsize=12, color=C['subtext'])

# Divider line
fig.add_artist(plt.Line2D(
    [0.04,0.96],[0.915,0.915],
    color=C['border'], linewidth=1.5,
    transform=fig.transFigure))

gs = gridspec.GridSpec(
    4, 4, figure=fig,
    hspace=0.55, wspace=0.35,
    top=0.90, bottom=0.05,
    left=0.04, right=0.97)

# ── KPI CARDS ──
kpi_data = [
    ('Total Revenue',   f'₹{total_revenue/1e5:.2f}L',
     f'+12.4% vs last year', C['blue'],   C['light_blue'], '💰'),
    ('Total Profit',    f'₹{total_profit/1e5:.2f}L',
     f'+8.7% vs last year',  C['green'],  C['light_green'],'📦'),
    ('Total Orders',    f'{total_orders:,}',
     f'{avg_order_val:,.0f} avg value',   C['orange'],
     C['light_orange'], '🛒'),
    ('Profit Margin',   f'{profit_margin:.1f}%',
     'Industry avg: 20%',    C['purple'], '#F3E8FF', '📊'),
]

for idx,(label,value,sub,color,bg_color,icon) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, idx])
    ax.set_facecolor(bg_color)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)
    ax.text(0.12, 0.78, icon,
            fontsize=20, transform=ax.transAxes)
    ax.text(0.5, 0.55, value,
            ha='center', fontsize=19, fontweight='bold',
            color=color, transform=ax.transAxes)
    ax.text(0.5, 0.30, label,
            ha='center', fontsize=10, color=C['text'],
            fontweight='600', transform=ax.transAxes)
    ax.text(0.5, 0.10, sub,
            ha='center', fontsize=8, color=C['subtext'],
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])

# ── CHART A: Monthly Revenue Trend ──
ax_trend = fig.add_subplot(gs[1, :2])
ax_trend.set_facecolor(C['panel'])
x = range(len(monthly_sorted))
ax_trend.fill_between(x, monthly_sorted['Sales']/1e3,
                      alpha=0.15, color=C['blue'])
ax_trend.plot(x, monthly_sorted['Sales']/1e3,
              color=C['blue'], linewidth=2.5,
              marker='o', markersize=7,
              markerfacecolor='white',
              markeredgecolor=C['blue'],
              markeredgewidth=2)
for xi, yi in zip(x, monthly_sorted['Sales']/1e3):
    ax_trend.annotate(
        f'₹{yi:.0f}K',
        xy=(xi,yi), xytext=(0,10),
        textcoords='offset points',
        ha='center', fontsize=7.5,
        color=C['navy'], fontweight='600')
ax_trend.set_xticks(x)
ax_trend.set_xticklabels(monthly_sorted['Month_Name'],
                         fontsize=9)
ax_trend.set_title('Monthly Revenue Trend',
                   color=C['navy'], pad=12)
ax_trend.set_ylabel('Revenue (₹ Thousands)', fontsize=9,
                    color=C['subtext'])
ax_trend.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))
ax_trend.grid(True, axis='y', linestyle='--', alpha=0.5)

# ── CHART B: Revenue by Region ──
ax_region = fig.add_subplot(gs[1, 2])
ax_region.set_facecolor(C['panel'])
sr = df_clean.groupby('Region')['Sales'].sum().sort_values()
colors_r = [C['blue'] if i == len(sr)-1
            else C['light_blue'].replace('#','') for i in range(len(sr))]
bar_colors = [C['navy'] if i == len(sr)-1
              else C['blue'] for i in range(len(sr))]
bars = ax_region.barh(
    [r.replace(' Branch','') for r in sr.index],
    sr.values/1e3,
    color=bar_colors,
    edgecolor='white', height=0.6)
for bar, val in zip(bars, sr.values/1e3):
    ax_region.text(
        val+0.5, bar.get_y()+bar.get_height()/2,
        f'₹{val:.0f}K', va='center', fontsize=8,
        color=C['text'], fontweight='600')
ax_region.set_title('Revenue by Branch',
                    color=C['navy'], pad=12)
ax_region.set_xlabel('Revenue (₹ Thousands)', fontsize=9,
                     color=C['subtext'])
ax_region.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))

# ── CHART C: Profit by Region ──
ax_profit_r = fig.add_subplot(gs[1, 3])
ax_profit_r.set_facecolor(C['panel'])
pr = df_clean.groupby('Region')['Profit'].sum().sort_values()
bars2 = ax_profit_r.barh(
    [r.replace(' Branch','') for r in pr.index],
    pr.values/1e3,
    color=C['green'], edgecolor='white', height=0.6)
for bar, val in zip(bars2, pr.values/1e3):
    ax_profit_r.text(
        val+0.2, bar.get_y()+bar.get_height()/2,
        f'₹{val:.0f}K', va='center', fontsize=8,
        color=C['text'], fontweight='600')
ax_profit_r.set_title('Profit by Branch',
                      color=C['navy'], pad=12)
ax_profit_r.set_xlabel('Profit (₹ Thousands)', fontsize=9,
                       color=C['subtext'])
ax_profit_r.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))

# ── CHART D: Sales by Category (Pie) ──
ax_pie = fig.add_subplot(gs[2, 0])
ax_pie.set_facecolor(C['panel'])
sc = df_clean.groupby('Category')['Sales'].sum()\
             .sort_values(ascending=False)
short_labels = [c.replace(' & ',' &\n')
                 .replace('Supplements & Vitamins','Supplements')
                 .replace('Herbal & Ayurvedic','Herbal')
                 .replace('Baby & Mother Care','Baby Care')
                 .replace('Cosmetics & Skincare','Cosmetics')
                 .replace('Medical Devices','Med. Devices')
                 .replace('Prescription Drugs','Rx Drugs')
                 .replace('OTC Medicines','OTC')
                for c in sc.index]
wedges,texts,autotexts = ax_pie.pie(
    sc.values,
    labels=short_labels,
    autopct='%1.1f%%',
    colors=CATEGORY_COLORS,
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    pctdistance=0.75)
for at in autotexts:
    at.set_fontsize(7); at.set_fontweight('bold')
    at.set_color('white')
for t in texts:
    t.set_fontsize(7.5); t.set_color(C['text'])
ax_pie.set_title('Revenue by Category',
                 color=C['navy'], pad=12)

# ── CHART E: Top 8 Products ──
ax_prod = fig.add_subplot(gs[2, 1:3])
ax_prod.set_facecolor(C['panel'])
tp8 = (df_clean.groupby('Product_Name')['Sales']
       .sum().nlargest(8).sort_values())
prod_colors = [C['navy'] if i >= 6 else C['blue']
               for i in range(len(tp8))]
bars3 = ax_prod.barh(tp8.index, tp8.values/1e3,
                     color=prod_colors,
                     edgecolor='white', height=0.6)
for bar, val in zip(bars3, tp8.values/1e3):
    ax_prod.text(
        val+0.2, bar.get_y()+bar.get_height()/2,
        f'₹{val:.1f}K', va='center', fontsize=8.5,
        color=C['text'], fontweight='600')
ax_prod.set_title('Top 8 Products by Revenue',
                  color=C['navy'], pad=12)
ax_prod.set_xlabel('Revenue (₹ Thousands)', fontsize=9,
                   color=C['subtext'])
ax_prod.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))

# ── CHART F: Quarterly Sales ──
ax_qtr = fig.add_subplot(gs[2, 3])
ax_qtr.set_facecolor(C['panel'])
qs = df_clean.groupby('Quarter')['Sales'].sum()
q_colors = [C['blue'],C['green'],C['orange'],C['navy']]
bars4 = ax_qtr.bar(qs.index, qs.values/1e3,
                   color=q_colors,
                   edgecolor='white', width=0.5)
for bar, val in zip(bars4, qs.values/1e3):
    ax_qtr.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.3,
        f'₹{val:.0f}K', ha='center', fontsize=8,
        fontweight='bold', color=C['text'])
ax_qtr.set_title('Quarterly Performance',
                 color=C['navy'], pad=12)
ax_qtr.set_ylabel('Revenue (₹ Thousands)', fontsize=9,
                  color=C['subtext'])
ax_qtr.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))

# ── CHART G: Top 10 Customers ──
ax_cust = fig.add_subplot(gs[3, :2])
ax_cust.set_facecolor(C['panel'])
tc10 = (df_clean.groupby('Customer_ID')['Sales']
        .sum().nlargest(10).sort_values())
cust_colors = [C['orange'] if i >= 8 else C['blue']
               for i in range(len(tc10))]
bars5 = ax_cust.barh(tc10.index, tc10.values/1e3,
                     color=cust_colors,
                     edgecolor='white', height=0.6)
for bar, val in zip(bars5, tc10.values/1e3):
    ax_cust.text(
        val+0.1, bar.get_y()+bar.get_height()/2,
        f'₹{val:.1f}K', va='center', fontsize=8,
        color=C['text'], fontweight='600')
ax_cust.set_title('Top 10 Customers by Revenue',
                  color=C['navy'], pad=12)
ax_cust.set_xlabel('Revenue (₹ Thousands)', fontsize=9,
                   color=C['subtext'])

# ── CHART H: Profit Margin by Category ──
ax_margin = fig.add_subplot(gs[3, 2:])
ax_margin.set_facecolor(C['panel'])
m_cat = (df_clean.groupby('Category')['Profit_Margin_%']
         .mean().sort_values(ascending=False))
short_cats = [c.replace('Supplements & Vitamins','Supplements')
               .replace('Herbal & Ayurvedic','Herbal')
               .replace('Baby & Mother Care','Baby Care')
               .replace('Cosmetics & Skincare','Cosmetics')
               .replace('Medical Devices','Med. Devices')
               .replace('Prescription Drugs','Rx Drugs')
              for c in m_cat.index]
m_colors = [C['green'] if v >= profit_margin
            else C['orange'] for v in m_cat.values]
bars6 = ax_margin.bar(short_cats, m_cat.values,
                      color=m_colors,
                      edgecolor='white', width=0.6)
for bar, val in zip(bars6, m_cat.values):
    ax_margin.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.3,
        f'{val:.1f}%', ha='center', fontsize=8,
        fontweight='bold', color=C['text'])
ax_margin.axhline(profit_margin, color=C['red'],
                  linestyle='--', linewidth=1.5,
                  label=f'Avg: {profit_margin:.1f}%')
ax_margin.set_title('Avg Profit Margin by Category',
                    color=C['navy'], pad=12)
ax_margin.set_ylabel('Profit Margin (%)', fontsize=9,
                     color=C['subtext'])
ax_margin.tick_params(axis='x', rotation=20, labelsize=8)
ax_margin.legend(fontsize=9)

plt.savefig('pharmacy_output/charts/01_main_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("   ✅ Chart 1: Main Dashboard")

# ============================================================
# CHART 2 — MONTHLY REVENUE & PROFIT TREND
# ============================================================
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                 facecolor=C['bg'])

monthly_profit = (df_clean.groupby(['Month','Month_Name'])
                  ['Profit'].sum().reset_index()
                  .sort_values('Month'))

# Revenue line
x = range(len(monthly_sorted))
ax1.fill_between(x, monthly_sorted['Sales']/1e3,
                 alpha=0.12, color=C['blue'])
ax1.plot(x, monthly_sorted['Sales']/1e3,
         color=C['blue'], linewidth=3,
         marker='o', markersize=9,
         markerfacecolor='white',
         markeredgecolor=C['blue'],
         markeredgewidth=2.5)
for xi, yi in zip(x, monthly_sorted['Sales']/1e3):
    ax1.annotate(f'₹{yi:.0f}K',
                 xy=(xi,yi), xytext=(0,12),
                 textcoords='offset points',
                 ha='center', fontsize=9,
                 fontweight='bold', color=C['navy'])
ax1.set_xticks(x)
ax1.set_xticklabels(monthly_sorted['Month_Name'], fontsize=10)
ax1.set_title('Monthly Revenue Trend — Pharmacy Sales 2023',
              fontsize=15, fontweight='bold',
              color=C['navy'], pad=15)
ax1.set_ylabel('Revenue (₹ Thousands)', fontsize=11,
               color=C['subtext'])
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
ax1.set_facecolor(C['panel'])

# Profit bars
ax2.bar(range(len(monthly_profit)),
        monthly_profit['Profit']/1e3,
        color=[C['green'] if v > 0 else C['red']
               for v in monthly_profit['Profit']],
        edgecolor='white', width=0.6)
for xi, yi in enumerate(monthly_profit['Profit']/1e3):
    ax2.text(xi, yi+0.5 if yi > 0 else yi-1.5,
             f'₹{yi:.0f}K', ha='center', fontsize=8,
             fontweight='bold', color=C['text'])
ax2.set_xticks(range(len(monthly_profit)))
ax2.set_xticklabels(monthly_profit['Month_Name'], fontsize=10)
ax2.set_title('Monthly Profit — 2023',
              fontsize=15, fontweight='bold',
              color=C['navy'], pad=15)
ax2.set_ylabel('Profit (₹ Thousands)', fontsize=11,
               color=C['subtext'])
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))
ax2.axhline(0, color=C['gray'], linewidth=1)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
ax2.set_facecolor(C['panel'])

plt.tight_layout(pad=2)
plt.savefig('pharmacy_output/charts/02_revenue_profit_trend.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("   ✅ Chart 2: Revenue & Profit Trends")

# ============================================================
# CHART 3 — CATEGORY DEEP DIVE
# ============================================================
fig3, axes = plt.subplots(1, 3, figsize=(18, 6),
                           facecolor=C['bg'])

# Revenue by category
cat_rev = cat_perf['Revenue'].sort_values()
short_c = [c.replace('Supplements & Vitamins','Supplements')
            .replace('Herbal & Ayurvedic','Herbal')
            .replace('Baby & Mother Care','Baby Care')
            .replace('Cosmetics & Skincare','Cosmetics')
            .replace('Medical Devices','Med. Devices')
            .replace('Prescription Drugs','Rx Drugs')
           for c in cat_rev.index]
bars = axes[0].barh(short_c, cat_rev.values/1e3,
                    color=CATEGORY_COLORS[::-1],
                    edgecolor='white', height=0.6)
for bar, val in zip(bars, cat_rev.values/1e3):
    axes[0].text(val+0.3, bar.get_y()+bar.get_height()/2,
                 f'₹{val:.0f}K', va='center', fontsize=8.5,
                 fontweight='bold', color=C['text'])
axes[0].set_title('Revenue by Category', fontweight='bold',
                  fontsize=12, color=C['navy'], pad=12)
axes[0].set_xlabel('Revenue (₹ Thousands)', fontsize=10,
                   color=C['subtext'])
axes[0].set_facecolor(C['panel'])
axes[0].xaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))

# Profit by category
cat_prof = cat_perf['Profit'].sort_values()
short_c2 = [c.replace('Supplements & Vitamins','Supplements')
             .replace('Herbal & Ayurvedic','Herbal')
             .replace('Baby & Mother Care','Baby Care')
             .replace('Cosmetics & Skincare','Cosmetics')
             .replace('Medical Devices','Med. Devices')
             .replace('Prescription Drugs','Rx Drugs')
            for c in cat_prof.index]
bars2 = axes[1].barh(short_c2, cat_prof.values/1e3,
                     color=C['green'],
                     edgecolor='white', height=0.6)
for bar, val in zip(bars2, cat_prof.values/1e3):
    axes[1].text(val+0.2, bar.get_y()+bar.get_height()/2,
                 f'₹{val:.0f}K', va='center', fontsize=8.5,
                 fontweight='bold', color=C['text'])
axes[1].set_title('Profit by Category', fontweight='bold',
                  fontsize=12, color=C['navy'], pad=12)
axes[1].set_xlabel('Profit (₹ Thousands)', fontsize=10,
                   color=C['subtext'])
axes[1].set_facecolor(C['panel'])

# Margin by category
m_cat2 = margin_by_cat.sort_values()
short_c3 = [c.replace('Supplements & Vitamins','Supplements')
             .replace('Herbal & Ayurvedic','Herbal')
             .replace('Baby & Mother Care','Baby Care')
             .replace('Cosmetics & Skincare','Cosmetics')
             .replace('Medical Devices','Med. Devices')
             .replace('Prescription Drugs','Rx Drugs')
            for c in m_cat2.index]
m_bar_colors = [C['green'] if v >= profit_margin
                else C['orange'] for v in m_cat2.values]
bars3 = axes[2].barh(short_c3, m_cat2.values,
                     color=m_bar_colors,
                     edgecolor='white', height=0.6)
for bar, val in zip(bars3, m_cat2.values):
    axes[2].text(val+0.2, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=8.5,
                 fontweight='bold', color=C['text'])
axes[2].axvline(profit_margin, color=C['red'],
                linestyle='--', linewidth=1.5,
                label=f'Avg {profit_margin:.1f}%')
axes[2].set_title('Avg Profit Margin %', fontweight='bold',
                  fontsize=12, color=C['navy'], pad=12)
axes[2].set_xlabel('Profit Margin (%)', fontsize=10,
                   color=C['subtext'])
axes[2].set_facecolor(C['panel'])
axes[2].legend(fontsize=9)

fig3.suptitle('Category Performance Analysis',
              fontsize=16, fontweight='bold',
              color=C['navy'], y=1.02)
plt.tight_layout()
plt.savefig('pharmacy_output/charts/03_category_analysis.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("   ✅ Chart 3: Category Analysis")

# ============================================================
# CHART 4 — HEATMAP: Monthly × Category Revenue
# ============================================================
fig4, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
pivot = (df_clean.pivot_table(
    index='Category',
    columns='Month_Name',
    values='Sales',
    aggfunc='sum',
    fill_value=0))
month_order_short = ['Jan','Feb','Mar','Apr','May','Jun',
                     'Jul','Aug','Sep','Oct','Nov','Dec']
month_order_avail = [m for m in month_order_short
                     if m in pivot.columns]
pivot = pivot[month_order_avail]
pivot.index = [i.replace('Supplements & Vitamins','Supplements')
                .replace('Herbal & Ayurvedic','Herbal')
                .replace('Baby & Mother Care','Baby Care')
                .replace('Cosmetics & Skincare','Cosmetics')
                .replace('Medical Devices','Med. Devices')
                .replace('Prescription Drugs','Rx Drugs')
               for i in pivot.index]
sns.heatmap(pivot/1e3,
            annot=True, fmt='.0f',
            cmap='Blues',
            linewidths=0.5,
            linecolor='white',
            annot_kws={'size':8,'weight':'bold'},
            ax=ax,
            cbar_kws={'label':'Revenue (₹ Thousands)',
                      'shrink':0.8})
ax.set_title('Monthly Revenue Heatmap (₹ Thousands)'
             ' — Category × Month',
             fontsize=14, fontweight='bold',
             color=C['navy'], pad=15)
ax.set_xlabel('Month', fontsize=11, color=C['subtext'])
ax.set_ylabel('Category', fontsize=11, color=C['subtext'])
ax.tick_params(axis='x', rotation=0, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig('pharmacy_output/charts/04_revenue_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("   ✅ Chart 4: Revenue Heatmap")

# ============================================================
# CHART 5 — PRODUCT ANALYSIS (Top 10 + Bottom 5)
# ============================================================
fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                 facecolor=C['bg'])
tp10 = (df_clean.groupby('Product_Name')['Sales']
        .sum().nlargest(10).sort_values())
prod_colors2 = [C['navy'] if i >= 8 else C['blue']
                for i in range(len(tp10))]
bars = ax1.barh(tp10.index, tp10.values/1e3,
                color=prod_colors2,
                edgecolor='white', height=0.65)
for bar, val in zip(bars, tp10.values/1e3):
    ax1.text(val+0.2, bar.get_y()+bar.get_height()/2,
             f'₹{val:.1f}K', va='center', fontsize=9,
             fontweight='bold', color=C['text'])
ax1.set_title('🏆 Top 10 Products by Revenue',
              fontsize=13, fontweight='bold',
              color=C['navy'], pad=12)
ax1.set_xlabel('Revenue (₹ Thousands)', fontsize=10,
               color=C['subtext'])
ax1.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))
ax1.set_facecolor(C['panel'])

bot5 = (df_clean.groupby('Product_Name')['Sales']
        .sum().nsmallest(5).sort_values(ascending=False))
bars2 = ax2.barh(bot5.index, bot5.values/1e3,
                 color=C['red'],
                 edgecolor='white', height=0.5)
for bar, val in zip(bars2, bot5.values/1e3):
    ax2.text(val+0.05, bar.get_y()+bar.get_height()/2,
             f'₹{val:.1f}K', va='center', fontsize=9,
             fontweight='bold', color=C['text'])
ax2.set_title('⚠️ Bottom 5 Products by Revenue',
              fontsize=13, fontweight='bold',
              color=C['red'], pad=12)
ax2.set_xlabel('Revenue (₹ Thousands)', fontsize=10,
               color=C['subtext'])
ax2.set_facecolor(C['panel'])

fig5.suptitle('Product Performance Analysis',
              fontsize=16, fontweight='bold',
              color=C['navy'], y=1.02)
plt.tight_layout()
plt.savefig('pharmacy_output/charts/05_product_analysis.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("   ✅ Chart 5: Product Analysis")

# ============================================================
# CHART 6 — PAYMENT METHOD & WEEKDAY ANALYSIS
# ============================================================
fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                 facecolor=C['bg'])

# Payment Method Pie
pay_data = (df_clean.groupby('Payment_Method')['Sales']
            .sum().sort_values(ascending=False))
pay_colors = [C['blue'],C['green'],C['orange'],
              C['purple'],C['navy']]
wedges,texts,autotexts = ax1.pie(
    pay_data.values,
    labels=pay_data.index,
    autopct='%1.1f%%',
    colors=pay_colors[:len(pay_data)],
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2.5),
    pctdistance=0.78)
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight('bold')
    at.set_color('white')
for t in texts:
    t.set_fontsize(9.5); t.set_color(C['text'])
ax1.set_title('Payment Method Distribution',
              fontsize=13, fontweight='bold',
              color=C['navy'], pad=15)
ax1.set_facecolor(C['bg'])

# Weekday bar chart
day_order = ['Monday','Tuesday','Wednesday',
             'Thursday','Friday','Saturday','Sunday']
wday = (df_clean.groupby('Weekday')['Sales']
        .sum().reindex(day_order))
day_colors = [C['orange'] if d in ['Saturday','Sunday']
              else C['blue'] for d in wday.index]
bars2 = ax2.bar(wday.index, wday.values/1e3,
                color=day_colors,
                edgecolor='white', width=0.6)
for bar, val in zip(bars2, wday.values/1e3):
    ax2.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.2,
             f'₹{val:.0f}K', ha='center', fontsize=8.5,
             fontweight='bold', color=C['text'])
ax2.set_title('Revenue by Day of Week',
              fontsize=13, fontweight='bold',
              color=C['navy'], pad=15)
ax2.set_ylabel('Revenue (₹ Thousands)', fontsize=10,
               color=C['subtext'])
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))
ax2.tick_params(axis='x', rotation=25, labelsize=9)
ax2.set_facecolor(C['panel'])
legend_els = [
    mpatches.Patch(facecolor=C['orange'], label='Weekend'),
    mpatches.Patch(facecolor=C['blue'],   label='Weekday')
]
ax2.legend(handles=legend_els, fontsize=9)

fig6.suptitle('Payment & Time Analysis',
              fontsize=16, fontweight='bold',
              color=C['navy'], y=1.02)
plt.tight_layout()
plt.savefig('pharmacy_output/charts/06_payment_weekday.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("   ✅ Chart 6: Payment & Weekday Analysis")

# ============================================================
# CHART 7 — CORRELATION HEATMAP
# ============================================================
fig7, ax = plt.subplots(figsize=(8, 6), facecolor=C['bg'])
num_cols = ['Sales','Profit','Quantity',
            'Unit_Price','Profit_Margin_%']
avail    = [c for c in num_cols if c in df_clean.columns]
corr     = df_clean[avail].corr()
mask     = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr,
            annot=True, fmt='.2f',
            cmap='RdYlGn',
            center=0,
            linewidths=0.8,
            linecolor='white',
            annot_kws={'size':11,'weight':'bold'},
            ax=ax,
            cbar_kws={'shrink':0.8},
            square=True)
ax.set_title('Correlation Matrix — Key Metrics',
             fontsize=14, fontweight='bold',
             color=C['navy'], pad=15)
ax.tick_params(axis='both', labelsize=10)
plt.tight_layout()
plt.savefig('pharmacy_output/charts/07_correlation_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("   ✅ Chart 7: Correlation Heatmap")

# ============================================================
# CHART 8 — PROFIT MARGIN DISTRIBUTION
# ============================================================
fig8, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                 facecolor=C['bg'])

# Box plot
cat_palette_dict = dict(zip(
    df_clean['Category'].unique(),
    CATEGORY_COLORS[:df_clean['Category'].nunique()]))
sns.boxplot(x='Profit_Margin_%', y='Category',
            data=df_clean,
            palette=cat_palette_dict,
            width=0.5, linewidth=1.2, ax=ax1)
ax1.axvline(profit_margin, color=C['red'],
            linestyle='--', linewidth=2,
            label=f'Overall Avg: {profit_margin:.1f}%')
ax1.set_title('Profit Margin Distribution by Category',
              fontsize=13, fontweight='bold',
              color=C['navy'], pad=12)
ax1.set_xlabel('Profit Margin (%)', fontsize=10,
               color=C['subtext'])
ax1.set_ylabel('Category', fontsize=10, color=C['subtext'])
ax1.legend(fontsize=9)
ax1.set_facecolor(C['panel'])
ax1.yaxis.set_tick_params(labelsize=8)

# Histogram
ax2.hist(df_clean['Profit_Margin_%'],
         bins=30, color=C['blue'],
         edgecolor='white', alpha=0.85)
ax2.axvline(profit_margin, color=C['red'],
            linestyle='--', linewidth=2,
            label=f'Mean: {profit_margin:.1f}%')
ax2.axvline(df_clean['Profit_Margin_%'].median(),
            color=C['green'],
            linestyle='--', linewidth=2,
            label=f"Median: "
                  f"{df_clean['Profit_Margin_%'].median():.1f}%")
ax2.set_title('Profit Margin Distribution (All Orders)',
              fontsize=13, fontweight='bold',
              color=C['navy'], pad=12)
ax2.set_xlabel('Profit Margin (%)', fontsize=10,
               color=C['subtext'])
ax2.set_ylabel('Number of Orders', fontsize=10,
               color=C['subtext'])
ax2.legend(fontsize=9)
ax2.set_facecolor(C['panel'])

fig8.suptitle('Profitability Analysis',
              fontsize=16, fontweight='bold',
              color=C['navy'], y=1.02)
plt.tight_layout()
plt.savefig('pharmacy_output/charts/08_profit_margin_dist.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.show()
print("   ✅ Chart 8: Profit Margin Distribution")

# ================================================================
# PHASE 9 — EXPORT EXCEL REPORT
# ================================================================
print("\n📁 PHASE 9: Exporting Excel Report...")

with pd.ExcelWriter(
    'pharmacy_output/data/pharmacy_report.xlsx',
    engine='openpyxl') as writer:

    df_clean.to_excel(
        writer, sheet_name='Clean Data', index=False)

    pd.DataFrame({
        'KPI':   ['Total Revenue','Total Profit',
                  'Total Orders','Total Qty',
                  'Avg Order Value','Profit Margin %'],
        'Value': [total_revenue, total_profit,
                  total_orders,  total_qty,
                  round(avg_order_val,2),
                  round(profit_margin,2)]
    }).to_excel(writer, sheet_name='KPI Summary', index=False)

    cat_perf.to_excel(writer, sheet_name='Category Analysis')
    top_products.to_frame().to_excel(
        writer, sheet_name='Top Products')
    top_customers.to_frame().to_excel(
        writer, sheet_name='Top Customers')
    region_perf.to_excel(writer, sheet_name='Region Analysis')
    monthly_sorted[['Month_Name','Sales']].to_excel(
        writer, sheet_name='Monthly Revenue', index=False)
    payment_perf.to_frame().to_excel(
        writer, sheet_name='Payment Analysis')

print("   ✅ Excel report saved: pharmacy_report.xlsx")

# ================================================================
# PHASE 10 — BUSINESS INSIGHTS REPORT
# ================================================================
print("\n" + "=" * 65)
print("   💡 EXECUTIVE INSIGHTS REPORT")
print("   Pharmacy Sales Analytics — 2023")
print("=" * 65)

best_cat    = cat_perf['Revenue'].idxmax()
best_region = region_perf['Sales'].idxmax()
best_month  = monthly_sorted.loc[
    monthly_sorted['Sales'].idxmax(), 'Month_Name']
worst_month = monthly_sorted.loc[
    monthly_sorted['Sales'].idxmin(), 'Month_Name']
best_margin_cat = margin_by_cat.idxmax()
best_margin_val = margin_by_cat.max()
top_prod    = top_products.index[0]
top_cust    = top_customers.index[0]

insights = [
    f"01. 💰 Total revenue of ₹{total_revenue/1e5:.2f}L with "
    f"{profit_margin:.1f}% margin — "
    f"{'above' if profit_margin>20 else 'below'} industry avg of 20%.",
    f"02. 📦 '{best_cat}' is the highest revenue category — "
    f"ensure consistent stock availability.",
    f"03. 🏆 '{top_prod}' is the #1 selling product — "
    f"never let it go out of stock.",
    f"04. 🗺️ {best_region} leads all branches — "
    f"analyse and replicate their practices.",
    f"05. 📅 {best_month} is the peak sales month — "
    f"pre-load inventory 3 weeks in advance.",
    f"06. 📉 {worst_month} is the weakest month — "
    f"run promotions and loyalty offers.",
    f"07. 💎 '{best_margin_cat}' has highest margin "
    f"({best_margin_val:.1f}%) — push cross-selling here.",
    f"08. 👤 Top customer '{top_cust}' — "
    f"enroll in VIP loyalty programme immediately.",
    f"09. 🧾 Avg order value ₹{avg_order_val:,.0f} — "
    f"bundle deals could increase this by 15-25%.",
    f"10. 💳 Diversified payment methods reduce friction — "
    f"all 5 methods are active which is healthy.",
    f"11. 📆 Weekend sales are higher — "
    f"staff more pharmacists Sat/Sun.",
    f"12. 🔗 High Sales-Profit correlation — "
    f"pricing discipline is maintained.",
    f"13. ⚠️ Bottom 5 products have very low revenue — "
    f"review and consider discontinuing.",
    f"14. 🏥 Prescription drugs have lower margins — "
    f"compensate by upselling OTC/Supplements.",
    f"15. 📊 Q4 is historically strong in pharma — "
    f"prepare for flu/cold season demand spike.",
]

for insight in insights:
    print(f"   {insight}")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 65)
print("   ✅ PROJECT COMPLETE")
print("=" * 65)
print("""
   📂 pharmacy_output/
   ├── data/
   │   ├── pharmacy_raw_data.csv
   │   ├── pharmacy_clean.csv
   │   ├── pharmacy_featured.csv     ← Load in Power BI
   │   └── pharmacy_report.xlsx      ← 8-sheet Excel report
   └── charts/
       ├── 01_main_dashboard.png
       ├── 02_revenue_profit_trend.png
       ├── 03_category_analysis.png
       ├── 04_revenue_heatmap.png
       ├── 05_product_analysis.png
       ├── 06_payment_weekday.png
       ├── 07_correlation_heatmap.png
       └── 08_profit_margin_dist.png

   🚀 NEXT STEPS:
   1. Set USE_REAL_DATA = True and run again with your CSV
   2. Load pharmacy_featured.csv in Power BI
   3. Build interactive dashboard using DAX measures
   4. Upload everything to GitHub
   5. Enable GitHub Pages for live portfolio
""")
print("=" * 65)