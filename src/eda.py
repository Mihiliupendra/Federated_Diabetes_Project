import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_raw_data(path):
    """Load raw UCI diabetes dataset"""
    print("Loading raw data...")
    df = pd.read_csv(path, na_values='?')
    return df

def analyze_missing_values(df):
    """Analyze missing value patterns"""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(missing_df[missing_df['Missing_Percentage'] > 0])
    return missing_df

def analyze_target_distribution(df):
    """Analyze readmission distribution"""
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    readmit_counts = df['readmitted'].value_counts()
    readmit_pct = df['readmitted'].value_counts(normalize=True) * 100
    
    print(f"Target Variable Distribution:")
    print(f"  <30 (Readmitted): {readmit_counts.get('<30', 0)} ({readmit_pct.get('<30', 0):.2f}%)")
    print(f"  >30 (Readmitted later): {readmit_counts.get('>30', 0)} ({readmit_pct.get('>30', 0):.2f}%)")
    print(f"  No (Not readmitted): {readmit_counts.get('No', 0)} ({readmit_pct.get('No', 0):.2f}%)")
    print(f"\nClass Imbalance: Positive class = {readmit_pct.get('<30', 0):.2f}%")
    print(f"Note: This is highly imbalanced (11% positive). Requires SMOTE/class_weight.")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    readmit_counts.plot(kind='bar', ax=axes[0], color=['skyblue', 'coral', 'lightgreen'])
    axes[0].set_title('Readmission Count Distribution')
    axes[0].set_ylabel('Number of Patients')
    
    readmit_pct.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
    axes[1].set_title('Readmission Percentage Distribution')
    
    plt.tight_layout()
    plt.savefig('results/01_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_demographics(df):
    """Analyze demographic distributions"""
    print("\n" + "="*60)
    print("DEMOGRAPHIC ANALYSIS")
    print("="*60)
    
    # Race
    print("\nRace Distribution:")
    print(df['race'].value_counts())
    
    # Gender
    print("\nGender Distribution:")
    print(df['gender'].value_counts())
    
    # Age
    print("\nAge Distribution:")
    print(df['age'].value_counts().sort_index())
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    df['race'].value_counts().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Race Distribution')
    axes[0].set_ylabel('Count')
    
    df['gender'].value_counts().plot(kind='bar', ax=axes[1])
    axes[1].set_title('Gender Distribution')
    axes[1].set_ylabel('Count')
    
    df['age'].value_counts().sort_index().plot(kind='bar', ax=axes[2])
    axes[2].set_title('Age Distribution')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('results/02_demographics.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_readmission_by_demographics(df):
    """Analyze readmission rates by demographic groups"""
    print("\n" + "="*60)
    print("READMISSION BY DEMOGRAPHICS (For Fairness Baseline)")
    print("="*60)
    
    # Create binary readmission
    df_temp = df.copy()
    df_temp['readmitted_binary'] = (df_temp['readmitted'] == '<30').astype(int)
    
    # By race
    print("\nReadmission Rate by Race:")
    race_readmit = df_temp.groupby('race')['readmitted_binary'].agg(['sum', 'count', 'mean'])
    race_readmit.columns = ['Readmitted_Count', 'Total', 'Readmission_Rate']
    print(race_readmit)
    
    # By gender
    print("\nReadmission Rate by Gender:")
    gender_readmit = df_temp.groupby('gender')['readmitted_binary'].agg(['sum', 'count', 'mean'])
    gender_readmit.columns = ['Readmitted_Count', 'Total', 'Readmission_Rate']
    print(gender_readmit)
    
    # By age group
    print("\nReadmission Rate by Age Group:")
    age_readmit = df_temp.groupby('age')['readmitted_binary'].agg(['sum', 'count', 'mean'])
    age_readmit.columns = ['Readmitted_Count', 'Total', 'Readmission_Rate']
    print(age_readmit)
    
    # Visualization: Fairness gap baseline
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    race_readmit['Readmission_Rate'].plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Readmission Rate by Race')
    axes[0].set_ylabel('Readmission Rate')
    axes[0].axhline(y=df_temp['readmitted_binary'].mean(), color='r', linestyle='--', label='Overall')
    axes[0].legend()
    
    gender_readmit['Readmission_Rate'].plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Readmission Rate by Gender')
    axes[1].set_ylabel('Readmission Rate')
    axes[1].axhline(y=df_temp['readmitted_binary'].mean(), color='r', linestyle='--', label='Overall')
    axes[1].legend()
    
    age_readmit['Readmission_Rate'].plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Readmission Rate by Age')
    axes[2].set_ylabel('Readmission Rate')
    axes[2].axhline(y=df_temp['readmitted_binary'].mean(), color='r', linestyle='--', label='Overall')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/03_readmission_by_demographics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate fairness gap (BASELINE)
    print("\n" + "="*60)
    print("FAIRNESS GAP BASELINE (Before Model)")
    print("="*60)
    
    race_rates = race_readmit['Readmission_Rate'].values
    race_gap = race_rates.max() - race_rates.min()
    print(f"Race fairness gap: {race_gap:.4f} ({race_gap*100:.2f}%)")
    
    gender_rates = gender_readmit['Readmission_Rate'].values
    gender_gap = gender_rates.max() - gender_rates.min()
    print(f"Gender fairness gap: {gender_gap:.4f} ({gender_gap*100:.2f}%)")

def run_eda():
    """Run complete EDA"""
    # Fix: Ensure path is correct for your Windows setup
    df = load_raw_data('data/diabetic_data.csv')
    print(f"Loaded data: {df.shape}")
    
    analyze_missing_values(df)
    analyze_target_distribution(df)
    analyze_demographics(df)
    analyze_readmission_by_demographics(df)
    
    print("\n EDA Complete! Visualizations saved to results/")

if __name__ == "__main__":
    run_eda()