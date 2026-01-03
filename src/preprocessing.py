import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def load_data(path):
    """Load raw data with '?' as missing values"""
    print("Loading data...")
    try:
        df = pd.read_csv(path, na_values='?')
        print(f"✅ Data loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        print("   Make sure diabetic_data.csv is in data/ folder")
        raise

def group_diagnosis_codes(code):
    """Map ICD-9 codes to 9 clinical categories"""
    if pd.isna(code): 
        return 'Other'
    
    try:
        code_str = str(code).strip()
        
        # Handle supplementary codes
        if code_str.startswith('V') or code_str.startswith('E'):
            return 'Other'
        
        code_num = float(code_str)
        
        # Diagnosis mapping
        if 390 <= code_num <= 459 or code_num == 785: 
            return 'Circulatory'
        elif 460 <= code_num <= 519 or code_num == 786: 
            return 'Respiratory'
        elif 520 <= code_num <= 579 or code_num == 787: 
            return 'Digestive'
        elif 250 <= code_num < 251: 
            return 'Diabetes'
        elif 800 <= code_num <= 999: 
            return 'Injury'
        elif 710 <= code_num <= 739: 
            return 'Musculoskeletal'
        elif 580 <= code_num <= 629 or code_num == 788: 
            return 'Genitourinary'
        elif 140 <= code_num <= 239: 
            return 'Neoplasms'
        else: 
            return 'Other'
    except (ValueError, AttributeError):
        return 'Other'

def create_hospital_ids(df):
    """Create 3 hospital client IDs based on age stratification"""
    
    age_mapping = {
        '[0, 10)': 5, '[10, 20)': 15, '[20, 30)': 25, '[30, 40)': 35,
        '[40, 50)': 45, '[50, 60)': 55, '[60, 70)': 65, '[70, 80)': 75,
        '[80, 90)': 85, '[90, 100)': 95
    }
    
    age_numeric = df['age'].map(age_mapping)
    hospital_id = np.zeros(len(df), dtype=int)
    
    # Hospital A: Geriatric (age >= 70)
    hospital_id[age_numeric >= 70] = 1
    
    # Hospital B: Adult (age 40-70)
    hospital_id[(age_numeric >= 40) & (age_numeric < 70)] = 2
    
    # Hospital C: Young (age < 40)
    hospital_id[age_numeric < 40] = 3
    
    return hospital_id

def preprocess_pipeline():
    """Complete preprocessing pipeline"""
    
    print("\n" + "="*70)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*70)
    
    # 1. Load Data
    df = load_data('data/diabetic_data.csv')
    original_shape = df.shape
    print(f"Original shape: {df.shape}")
    
    # 2. Drop high missing value columns
    drop_cols = ['weight', 'payer_code', 'medical_specialty', 'encounter_id']
    available_drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=available_drop_cols)
    print(f"After dropping high-missing cols: {df.shape}")

    # 3. Filter deceased patients
    if 'discharge_disposition_id' in df.columns:
        df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
    print(f"After removing deceased patients: {df.shape}")

    # 4. Feature Engineering: Diagnosis codes
    print("\nGrouping ICD-9 diagnosis codes...")
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].apply(group_diagnosis_codes)

    # 5. Target binarization
    if 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    print(f"\n📊 Class Distribution:")
    print(df['readmitted'].value_counts())
    print(f"Proportions:\n{df['readmitted'].value_counts(normalize=True)}\n")

    # 6. Create hospital IDs
    print("Creating hospital client IDs (age-stratified)...")
    df['hospital_id'] = create_hospital_ids(df)
    print(f"Hospital distribution:")
    print(df['hospital_id'].value_counts().sort_index())

    # 7. Handle missing demographics
    print("\nHandling missing demographics...")
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Unknown')
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace('Unknown/Invalid', 'Unknown')

    # 8. Preserve sensitive attributes
    print("Preserving sensitive attributes...")
    sensitive_cols = ['hospital_id', 'race', 'gender', 'age']
    sensitive_attrs = df[[col for col in sensitive_cols if col in df.columns]].copy()

    # 9. Define feature groups
    medication_features = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citogliptin', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'change'
    ]

    categorical_features = [
        'race', 'gender', 'age', 'admission_type_id', 
        'discharge_disposition_id', 'admission_source_id',
        'diag_1', 'diag_2', 'diag_3', 
        'max_glu_serum', 'A1Cresult', 'diabetesMed'
    ] + medication_features

    numerical_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ]

    # Filter to only existing columns
    categorical_features = [col for col in categorical_features if col in df.columns]
    numerical_features = [col for col in numerical_features if col in df.columns]

    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")

    # 10. Prepare X and y
    X = df.drop(['readmitted', 'patient_nbr', 'hospital_id'], axis=1, errors='ignore')
    y = df['readmitted']

    # 11. Build preprocessing pipeline
    print("\nBuilding preprocessing pipeline...")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # 12. Fit and transform
    print("Fitting preprocessing pipeline...")
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)

    print(f"Processed features: {X_processed.shape[1]}")

    # 13. Train-test split
    print("\nPerforming stratified train-test split (80-20)...")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_processed, y, np.arange(len(X_processed)),
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    sens_train = sensitive_attrs.iloc[idx_train].reset_index(drop=True)
    sens_test = sensitive_attrs.iloc[idx_test].reset_index(drop=True)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training readmission rate: {y_train.mean():.4f}")
    print(f"Test readmission rate: {y_test.mean():.4f}")

    # 14. Handle class imbalance
    print("\nHandling class imbalance...")
    try:
        # Only apply SMOTE if minority class has enough samples
        minority_count = (y_train == 1).sum()
        if minority_count > 10:
            smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"✅ SMOTE applied successfully")
            print(f"  Training set: {X_train_balanced.shape}")
            print(f"  Class 0: {(y_train_balanced == 0).sum()}")
            print(f"  Class 1: {(y_train_balanced == 1).sum()}")
        else:
            print(f"⚠️ Not enough minority samples for SMOTE")
            X_train_balanced = X_train
            y_train_balanced = y_train
    except Exception as e:
        print(f"⚠️ SMOTE failed: {e}")
        print("   Using original unbalanced data")
        X_train_balanced = X_train
        y_train_balanced = y_train

    # 15. Hospital statistics
    print("\n" + "="*70)
    print("HOSPITAL STATISTICS (FOR NON-IID ANALYSIS)")
    print("="*70)
    
    hospital_stats = {}
    for hospital in [1, 2, 3]:
        mask = sens_test['hospital_id'] == hospital
        n_samples = mask.sum()
        if n_samples > 0:
            readmission_rate = y_test[mask].mean()
            hospital_stats[hospital] = {
                'n_samples': int(n_samples),
                'readmission_rate': float(readmission_rate),
                'class_distribution': {
                    'no_readmit': int((y_test[mask] == 0).sum()),
                    'readmit': int((y_test[mask] == 1).sum())
                }
            }
            
            print(f"\nHospital {hospital}:")
            print(f"  Records: {n_samples}")
            print(f"  Readmission rate: {readmission_rate:.4f} ({readmission_rate*100:.2f}%)")
            print(f"  No readmit: {hospital_stats[hospital]['class_distribution']['no_readmit']}")
            print(f"  Readmit <30: {hospital_stats[hospital]['class_distribution']['readmit']}")

    # 16. Save everything
    print("\n" + "="*70)
    print("SAVING PROCESSED DATA")
    print("="*70)

    output_dir = Path('data/processed_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'X_train.npy', X_train_balanced.astype(np.float32))
    np.save(output_dir / 'X_test.npy', X_test.astype(np.float32))
    np.save(output_dir / 'y_train.npy', y_train_balanced.astype(np.float32))
    np.save(output_dir / 'y_test.npy', y_test.astype(np.float32))

    sens_train.to_csv(output_dir / 'sensitive_attrs_train.csv', index=False)
    sens_test.to_csv(output_dir / 'sensitive_attrs_test.csv', index=False)

    with open(output_dir / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))

    with open(output_dir / 'hospital_stats.json', 'w') as f:
        json.dump(hospital_stats, f, indent=2)

    with open(output_dir / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print("✅ All data saved to:", output_dir)
    print(f"\nFiles created:")
    print(f"  • X_train.npy ({X_train_balanced.shape})")
    print(f"  • X_test.npy ({X_test.shape})")
    print(f"  • y_train.npy ({y_train_balanced.shape})")
    print(f"  • y_test.npy ({y_test.shape})")
    print(f"  • sensitive_attrs_train.csv")
    print(f"  • sensitive_attrs_test.csv")
    print(f"  • feature_names.txt ({len(feature_names)} features)")
    print(f"  • hospital_stats.json")
    print(f"  • preprocessor.pkl")
    
    print("\n" + "="*70)
    print("✅ PHASE 2 PREPROCESSING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    preprocess_pipeline()