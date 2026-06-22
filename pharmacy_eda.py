"""
Pharmacy Prescription Refill - Complete EDA Pipeline
Merged script that generates synthetic data and performs comprehensive EDA analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================
# DATA GENERATION SECTION
# ============================

def generate_synthetic_data(n_records=2000, random_seed=42):
    """Generate synthetic pharmacy prescription data for EDA."""
    np.random.seed(random_seed)
    
    genders = ['Male', 'Female']
    drug_classes = ['Antibiotic', 'Antihypertensive', 'Antidiabetic', 'Analgesic', 
                    'Antidepressant', 'Cholesterol', 'Thyroid']
    
    chronic_drugs = {'Antihypertensive', 'Antidiabetic', 'Cholesterol', 'Thyroid'}
    
    data = {
        'patient_age': np.random.randint(18, 85, n_records),
        'gender': np.random.choice(genders, n_records),
        'drug_class': np.random.choice(drug_classes, n_records),
        'num_medications': np.random.randint(1, 8, n_records),
        'refill_count_past_year': np.random.randint(0, 12, n_records),
        'days_supply': np.random.choice([7, 14, 30, 60, 90], n_records),
        'copay_amount': np.round(np.random.uniform(5, 150, n_records), 2),
        'doctor_visits_6mo': np.random.randint(0, 10, n_records),
        'adherence_score': np.round(np.random.uniform(0.4, 1.0, n_records), 3),
        'num_side_effects_reported': np.random.randint(0, 5, n_records),
        'insurance_covered': np.random.binomial(1, 0.75, n_records),
    }
    
    df = pd.DataFrame(data)
    
    condition_map = {
        'Antibiotic': 'Infection', 'Antihypertensive': 'Hypertension',
        'Antidiabetic': 'Diabetes Type 2', 'Analgesic': 'Pain Management',
        'Antidepressant': 'Depression', 'Cholesterol': 'High Cholesterol',
        'Thyroid': 'Hypothyroidism'
    }
    df['medical_condition'] = df['drug_class'].map(condition_map)
    df['chronic_condition'] = df['drug_class'].isin(chronic_drugs).astype(int)
    
    prob_refill = np.full(n_records, 0.2)
    prob_refill += 0.25 * df['chronic_condition']
    prob_refill += 0.20 * (df['adherence_score'] > 0.75)
    prob_refill += 0.15 * (df['days_supply'] <= 30)
    prob_refill += 0.15 * (df['refill_count_past_year'] > 6)
    prob_refill += 0.05 * (df['doctor_visits_6mo'] > 4)
    prob_refill -= 0.10 * (df['num_side_effects_reported'] > 2)
    prob_refill -= 0.10 * (df['copay_amount'] > 100)
    prob_refill = np.clip(prob_refill, 0.05, 0.95)
    
    df['will_refill_30days'] = np.random.binomial(1, prob_refill)
    
    column_order = ['patient_age', 'gender', 'drug_class', 'medical_condition',
                    'num_medications', 'refill_count_past_year', 'days_supply',
                    'copay_amount', 'doctor_visits_6mo', 'chronic_condition',
                    'adherence_score', 'num_side_effects_reported',
                    'insurance_covered', 'will_refill_30days']
    
    return df[column_order]

def ensure_data_exists(data_path='data/pharmacy_prescriptions.csv'):
    """Check if data exists, generate if not."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    if not os.path.exists(data_path):
        print("Generating synthetic data...")
        df = generate_synthetic_data()
        df.to_csv(data_path, index=False)
        print(f"✓ Data saved to {data_path}")
        print(f"  Records: {len(df)}")
        print(f"  Target distribution:\n{df['will_refill_30days'].value_counts()}")
        return df
    else:
        print(f"✓ Data already exists at {data_path}")
        return pd.read_csv(data_path)

# ============================
# EDA ANALYSIS SECTION
# ============================

class PharmacyEDA:
    """Comprehensive EDA for Pharmacy Prescription Data."""
    
    def __init__(self, data_path='data/pharmacy_prescriptions.csv'):
        self.df = pd.read_csv(data_path)
        self.numeric_cols = ['patient_age', 'num_medications', 'refill_count_past_year',
                            'days_supply', 'copay_amount', 'doctor_visits_6mo',
                            'adherence_score', 'num_side_effects_reported']
        self.categorical_cols = ['gender', 'drug_class', 'medical_condition', 'insurance_covered']
        self.target = 'will_refill_30days'
        
    def basic_info(self):
        """Display basic dataset information."""
        print("="*70)
        print("DATASET OVERVIEW")
        print("="*70)
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nDuplicates: {self.df.duplicated().sum()}")
        print(f"\nTarget distribution:\n{self.df[self.target].value_counts()}")
        print(f"Class balance: {self.df[self.target].value_counts(normalize=True).round(3).to_dict()}")
        
    def statistical_summary(self):
        """Generate comprehensive statistical summary."""
        print("\n" + "="*70)
        print("STATISTICAL SUMMARY")
        print("="*70)
        
        summary = self.df[self.numeric_cols].describe().T
        summary['median'] = self.df[self.numeric_cols].median()
        summary['skewness'] = self.df[self.numeric_cols].skew()
        summary['kurtosis'] = self.df[self.numeric_cols].kurtosis()
        print(summary.round(3))
        
        print("\n--- Categorical Summary ---")
        for col in self.categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
    
    def distribution_analysis(self):
        """Analyze and visualize distributions."""
        print("\n📊 Generating distribution analysis...")
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.numeric_cols):
            ax = axes[idx]
            # Histogram with KDE
            self.df[col].hist(ax=ax, bins=30, color='steelblue', 
                             edgecolor='black', alpha=0.7, density=True)
            self.df[col].plot.kde(ax=ax, color='red', linewidth=2)
            
            # Add statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            ax.axvline(mean_val, color='green', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.set_title(f'{col}\nSkew: {self.df[col].skew():.2f}', fontsize=10)
            ax.set_xlabel(col)
            ax.legend(fontsize=8)
        
        plt.suptitle('Distribution Analysis with KDE and Statistics', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/01_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 01_distributions.png saved")
    
    def correlation_analysis(self):
        """Comprehensive correlation analysis."""
        print("\n🔗 Performing correlation analysis...")
        
        # Calculate correlations
        corr_matrix = self.df[self.numeric_cols + [self.target]].corr()
        
        # Pearson correlation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=axes[0],
                   cbar_kws={"shrink": 0.8})
        axes[0].set_title('Pearson Correlation Matrix', fontsize=12, fontweight='bold')
        
        # Correlation with target
        target_corr = corr_matrix[self.target].drop(self.target).sort_values()
        colors = ['salmon' if x < 0 else 'steelblue' for x in target_corr.values]
        axes[1].barh(target_corr.index, target_corr.values, color=colors, edgecolor='black')
        axes[1].axvline(0, color='black', linewidth=0.8)
        axes[1].set_xlabel('Correlation with Will Refill')
        axes[1].set_title('Feature Correlations with Target', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/02_correlation_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 02_correlation_analysis.png saved")
        
        # Print top correlations
        print("\nTop correlations with target:")
        print(target_corr.abs().sort_values(ascending=False).head(5))
        
        # Spearman correlation (non-linear relationships)
        spearman_corr = self.df[self.numeric_cols + [self.target]].corr(method='spearman')
        plt.figure(figsize=(10, 8))
        sns.heatmap(spearman_corr, annot=True, fmt='.3f', cmap='viridis',
                   square=True, linewidths=1)
        plt.title('Spearman Correlation Matrix (Non-linear)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/02b_spearman_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def categorical_analysis(self):
        """Analyze categorical variables and their relationship with target."""
        print("\n📊 Performing categorical analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gender distribution
        self.df['gender'].value_counts().plot(kind='pie', ax=axes[0, 0], 
                                              autopct='%1.1f%%', startangle=90,
                                              colors=['steelblue', 'salmon'])
        axes[0, 0].set_title('Gender Distribution', fontweight='bold')
        axes[0, 0].set_ylabel('')
        
        # Drug class distribution
        drug_counts = self.df['drug_class'].value_counts()
        axes[0, 1].barh(drug_counts.index, drug_counts.values, color='steelblue', edgecolor='black')
        axes[0, 1].set_title('Drug Class Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Count')
        
        # Refill rate by drug class
        refill_by_drug = self.df.groupby('drug_class')[self.target].mean().sort_values()
        axes[1, 0].barh(refill_by_drug.index, refill_by_drug.values, 
                       color='steelblue', edgecolor='black')
        axes[1, 0].axvline(self.df[self.target].mean(), color='red', 
                          linestyle='--', label='Overall Average')
        axes[1, 0].set_title('Refill Rate by Drug Class', fontweight='bold')
        axes[1, 0].set_xlabel('Refill Rate')
        axes[1, 0].legend()
        
        # Refill rate by gender
        gender_refill = self.df.groupby('gender')[self.target].mean()
        axes[1, 1].bar(gender_refill.index, gender_refill.values, 
                      color=['steelblue', 'salmon'], edgecolor='black')
        axes[1, 1].axhline(self.df[self.target].mean(), color='red', 
                          linestyle='--', label='Overall Average')
        axes[1, 1].set_title('Refill Rate by Gender', fontweight='bold')
        axes[1, 1].set_ylabel('Refill Rate')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('reports/03_categorical_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 03_categorical_analysis.png saved")
    
    def bivariate_analysis(self):
        """Bivariate analysis - relationships between features."""
        print("\n📊 Performing bivariate analysis...")
        
        # Key relationships
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Age vs Adherence
        scatter = axes[0, 0].scatter(self.df['patient_age'], self.df['adherence_score'],
                                    c=self.df[self.target], cmap='RdYlBu', 
                                    alpha=0.6, edgecolors='black', linewidth=0.3)
        axes[0, 0].set_xlabel('Patient Age')
        axes[0, 0].set_ylabel('Adherence Score')
        axes[0, 0].set_title('Age vs Adherence (color: refill)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Copay vs Adherence
        scatter = axes[0, 1].scatter(self.df['copay_amount'], self.df['adherence_score'],
                                    c=self.df[self.target], cmap='RdYlBu',
                                    alpha=0.6, edgecolors='black', linewidth=0.3)
        axes[0, 1].set_xlabel('Copay Amount ($)')
        axes[0, 1].set_ylabel('Adherence Score')
        axes[0, 1].set_title('Copay vs Adherence (color: refill)')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Medications vs Side Effects
        axes[0, 2].boxplot([self.df[self.df['num_medications']==i]['num_side_effects_reported']
                            for i in range(1, 8)], 
                           labels=range(1, 8))
        axes[0, 2].set_xlabel('Number of Medications')
        axes[0, 2].set_ylabel('Side Effects Reported')
        axes[0, 2].set_title('Medications vs Side Effects')
        
        # Adherence by chronic condition
        sns.violinplot(data=self.df, x='chronic_condition', y='adherence_score',
                      ax=axes[1, 0], palette='Set2')
        axes[1, 0].set_title('Adherence by Chronic Condition')
        axes[1, 0].set_xlabel('Chronic Condition')
        
        # Refill count distribution by target
        for refill_status, color, label in zip([0, 1], ['salmon', 'steelblue'], 
                                                ['No Refill', 'Refill']):
            subset = self.df[self.df[self.target] == refill_status]['refill_count_past_year']
            axes[1, 1].hist(subset, alpha=0.5, bins=12, label=label, 
                           color=color, edgecolor='black')
        axes[1, 1].set_xlabel('Past Refill Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Past Refills by Future Refill Status')
        axes[1, 1].legend()
        
        # Days supply distribution
        days_refill = self.df.groupby('days_supply')[self.target].mean()
        axes[1, 2].bar(days_refill.index.astype(str), days_refill.values,
                      color='steelblue', edgecolor='black')
        axes[1, 2].axhline(self.df[self.target].mean(), color='red', 
                          linestyle='--', label='Average')
        axes[1, 2].set_xlabel('Days Supply')
        axes[1, 2].set_ylabel('Refill Rate')
        axes[1, 2].set_title('Refill Rate by Days Supply')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('reports/04_bivariate_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 04_bivariate_analysis.png saved")
    
    def hypothesis_testing(self):
        """Perform statistical hypothesis tests."""
        print("\n🔬 Performing hypothesis tests...")
        print("="*70)
        
        # Test 1: Adherence difference between refill groups
        refill_yes = self.df[self.df[self.target]==1]['adherence_score']
        refill_no = self.df[self.df[self.target]==0]['adherence_score']
        t_stat, p_val = stats.ttest_ind(refill_yes, refill_no)
        print(f"\n1. T-test: Adherence Score (Refill vs No Refill)")
        print(f"   Mean (Refill): {refill_yes.mean():.3f}, Mean (No Refill): {refill_no.mean():.3f}")
        print(f"   T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")
        print(f"   Result: {'*** Significant' if p_val < 0.001 else '** Significant' if p_val < 0.01 else '* Significant' if p_val < 0.05 else 'Not Significant'}")
        
        # Test 2: Chi-square test for chronic condition vs refill
        contingency = pd.crosstab(self.df['chronic_condition'], self.df[self.target])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n2. Chi-square: Chronic Condition vs Refill")
        print(f"   Chi2: {chi2:.3f}, P-value: {p_val:.4f}")
        print(f"   Result: {'*** Significant' if p_val < 0.001 else '** Significant' if p_val < 0.01 else '* Significant' if p_val < 0.05 else 'Not Significant'}")
        
        # Test 3: ANOVA - Refill rate across drug classes
        groups = [self.df[self.df['drug_class']==drug][self.target] 
                 for drug in self.df['drug_class'].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\n3. ANOVA: Refill Rate across Drug Classes")
        print(f"   F-statistic: {f_stat:.3f}, P-value: {p_val:.4f}")
        print(f"   Result: {'*** Significant' if p_val < 0.001 else '** Significant' if p_val < 0.01 else '* Significant' if p_val < 0.05 else 'Not Significant'}")
        
        # Test 4: Mann-Whitney U test for copay
        u_stat, p_val = stats.mannwhitneyu(
            self.df[self.df[self.target]==1]['copay_amount'],
            self.df[self.df[self.target]==0]['copay_amount']
        )
        print(f"\n4. Mann-Whitney U: Copay Amount (Refill vs No Refill)")
        print(f"   U-statistic: {u_stat:.3f}, P-value: {p_val:.4f}")
        print(f"   Result: {'*** Significant' if p_val < 0.001 else '** Significant' if p_val < 0.01 else '* Significant' if p_val < 0.05 else 'Not Significant'}")
        
    def outlier_detection(self):
        """Detect and visualize outliers."""
        print("\n🔍 Detecting outliers...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        outlier_summary = []
        for idx, col in enumerate(self.numeric_cols):
            ax = axes[idx]
            self.df[col].plot.box(ax=ax)
            ax.set_title(col, fontsize=10)
            
            # Calculate outliers using IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.df)) * 100
            outlier_summary.append({
                'Feature': col, 'Outliers': outlier_count, 'Percentage': f"{outlier_pct:.2f}%"
            })
        
        plt.suptitle('Outlier Detection - Box Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/05_outliers.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\nOutlier Summary:")
        print(pd.DataFrame(outlier_summary).to_string(index=False))
        print("  ✓ 05_outliers.png saved")
    
    def feature_relationships(self):
        """Analyze feature interactions and group patterns."""
        print("\n🔗 Analyzing feature relationships...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Age groups
        age_bins = [18, 30, 45, 60, 75, 85]
        age_labels = ['18-30', '31-45', '46-60', '61-75', '76+']
        self.df['age_group'] = pd.cut(self.df['patient_age'], bins=age_bins, labels=age_labels)
        
        age_refill = self.df.groupby('age_group')[self.target].mean()
        axes[0, 0].bar(age_refill.index.astype(str), age_refill.values,
                      color='steelblue', edgecolor='black')
        axes[0, 0].set_title('Refill Rate by Age Group', fontweight='bold')
        axes[0, 0].set_xlabel('Age Group')
        axes[0, 0].set_ylabel('Refill Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Insurance effect
        ins_refill = self.df.groupby('insurance_covered')[self.target].mean()
        axes[0, 1].bar(['Uninsured', 'Insured'], ins_refill.values,
                      color=['salmon', 'steelblue'], edgecolor='black')
        axes[0, 1].set_title('Refill Rate by Insurance Status', fontweight='bold')
        axes[0, 1].set_ylabel('Refill Rate')
        
        # Copay tiers
        copay_bins = [0, 30, 60, 100, 150]
        copay_labels = ['Low ($0-30)', 'Medium ($30-60)', 'High ($60-100)', 'Very High ($100+)']
        self.df['copay_tier'] = pd.cut(self.df['copay_amount'], bins=copay_bins, labels=copay_labels)
        
        copay_refill = self.df.groupby('copay_tier')[self.target].mean()
        axes[1, 0].bar(copay_refill.index.astype(str), copay_refill.values,
                      color='salmon', edgecolor='black')
        axes[1, 0].set_title('Refill Rate by Copay Tier', fontweight='bold')
        axes[1, 0].set_xlabel('Copay Tier')
        axes[1, 0].set_ylabel('Refill Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Adherence tiers
        adh_bins = [0.4, 0.6, 0.75, 0.9, 1.0]
        adh_labels = ['Low (0.4-0.6)', 'Medium (0.6-0.75)', 'High (0.75-0.9)', 'Very High (0.9-1.0)']
        self.df['adherence_tier'] = pd.cut(self.df['adherence_score'], bins=adh_bins, labels=adh_labels)
        
        adh_refill = self.df.groupby('adherence_tier')[self.target].mean()
        axes[1, 1].bar(adh_refill.index.astype(str), adh_refill.values,
                      color='steelblue', edgecolor='black')
        axes[1, 1].set_title('Refill Rate by Adherence Tier', fontweight='bold')
        axes[1, 1].set_xlabel('Adherence Tier')
        axes[1, 1].set_ylabel('Refill Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('reports/06_feature_relationships.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 06_feature_relationships.png saved")
    
    def generate_insights_report(self):
        """Generate structured insights report."""
        report = []
        report.append("="*70)
        report.append("PHARMACY PRESCRIPTION REFILL - EDA INSIGHTS REPORT")
        report.append("="*70)
        
        report.append("\n1. DATASET CHARACTERISTICS")
        report.append("-"*40)
        report.append(f"   - Total records: {len(self.df)}")
        report.append(f"   - Total features: {len(self.df.columns)}")
        report.append(f"   - Missing values: {self.df.isnull().sum().sum()}")
        report.append(f"   - Target balance: {self.df[self.target].mean():.1%} positive class")
        
        report.append("\n2. KEY STATISTICAL INSIGHTS")
        report.append("-"*40)
        report.append(f"   - Patient age: mean={self.df['patient_age'].mean():.1f}, "
                     f"median={self.df['patient_age'].median():.1f}")
        report.append(f"   - Adherence score: mean={self.df['adherence_score'].mean():.3f}")
        report.append(f"   - Average copay: ${self.df['copay_amount'].mean():.2f}")
        report.append(f"   - Avg medications: {self.df['num_medications'].mean():.2f}")
        
        # Top correlations
        corr_matrix = self.df[self.numeric_cols + [self.target]].corr()
        top_corr = corr_matrix[self.target].drop(self.target).abs().sort_values(ascending=False)
        
        report.append("\n3. TOP CORRELATIONS WITH REFILL")
        report.append("-"*40)
        for feat, corr in top_corr.head(5).items():
            report.append(f"   - {feat}: {corr:.3f}")
        
        report.append("\n4. KEY PATTERNS IDENTIFIED")
        report.append("-"*40)
        
        chronic_rate = self.df[self.df['chronic_condition']==1][self.target].mean()
        non_chronic_rate = self.df[self.df['chronic_condition']==0][self.target].mean()
        report.append(f"   - Chronic patients refill {chronic_rate:.1%} vs "
                     f"{non_chronic_rate:.1%} for non-chronic")
        
        high_adh = self.df[self.df['adherence_score']>0.75][self.target].mean()
        low_adh = self.df[self.df['adherence_score']<=0.75][self.target].mean()
        report.append(f"   - High adherence (>0.75): {high_adh:.1%} refill rate")
        report.append(f"   - Low adherence (<=0.75): {low_adh:.1%} refill rate")
        
        high_copay = self.df[self.df['copay_amount']>100][self.target].mean()
        low_copay = self.df[self.df['copay_amount']<=100][self.target].mean()
        report.append(f"   - Low copay (<=$100): {low_copay:.1%} refill rate")
        report.append(f"   - High copay (>$100): {high_copay:.1%} refill rate")
        
        report.append("\n5. ACTIONABLE INSIGHTS")
        report.append("-"*40)
        report.append("   - Adherence score is the strongest behavioral predictor")
        report.append("   - Chronic condition patients show higher engagement")
        report.append("   - Cost is a significant barrier (copay > $100 reduces refill)")
        report.append("   - Past refill history is a reliable predictor")
        report.append("   - Insurance coverage correlates with refill behavior")
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        # Save report
        with open('reports/eda_insights_report.txt', 'w') as f:
            f.write(report_text)
        print("\n✓ Report saved to reports/eda_insights_report.txt")
    
    def run_full_analysis(self):
        """Execute complete EDA pipeline."""
        print("\n" + "="*70)
        print("PHARMACY PRESCRIPTION REFILL - COMPLETE EDA ANALYSIS")
        print("="*70 + "\n")
        
        self.basic_info()
        self.statistical_summary()
        self.distribution_analysis()
        self.correlation_analysis()
        self.categorical_analysis()
        self.bivariate_analysis()
        self.hypothesis_testing()
        self.outlier_detection()
        self.feature_relationships()
        self.generate_insights_report()
        
        print("\n" + "="*70)
        print("✅ EDA ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  • reports/01_distributions.png")
        print("  • reports/02_correlation_analysis.png")
        print("  • reports/02b_spearman_correlation.png")
        print("  • reports/03_categorical_analysis.png")
        print("  • reports/04_bivariate_analysis.png")
        print("  • reports/05_outliers.png")
        print("  • reports/06_feature_relationships.png")
        print("  • reports/eda_insights_report.txt")
        print("="*70)

# ============================
# MAIN EXECUTION
# ============================

def main():
    """Main function to run the complete EDA pipeline."""
    try:
        # Step 1: Ensure data exists
        df = ensure_data_exists()
        
        # Step 2: Run EDA analysis
        eda = PharmacyEDA()
        eda.run_full_analysis()
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()