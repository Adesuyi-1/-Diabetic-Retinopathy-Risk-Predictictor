import pandas as pd
import numpy as np
import os

def generate_synthetic_dr_data(n_samples=10000):
    """Generate synthetic diabetic retinopathy risk dataset"""
    np.random.seed(42)
    
    # Clinical features based on real risk factors
    data = {
        'age': np.random.normal(58, 12, n_samples).astype(int),
        'gender': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
        'bmi': np.random.normal(28.5, 5, n_samples),
        'hbA1c': np.random.normal(7.8, 1.8, n_samples),
        'systolic_bp': np.random.normal(135, 20, n_samples),
        'diastolic_bp': np.random.normal(85, 12, n_samples),
        'years_with_diabetes': np.random.exponential(8, n_samples),
        'smoking_status': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),  # 0: Never, 1: Former, 2: Current
        'hypertension': np.random.binomial(1, 0.45, n_samples),
        'dyslipidemia': np.random.binomial(1, 0.38, n_samples),
        'family_history_dr': np.random.binomial(1, 0.15, n_samples),
        'serum_cholesterol': np.random.normal(195, 35, n_samples),
        'retinal_screening_regularity': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])  # 0: Irregular, 1: Annual, 2: Biannual
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target variable (risk of referable DR)
    # Based on clinical formula: risk increases with HbA1c, years_with_diabetes, hypertension
    risk_score = (
        0.3 * (df['hbA1c'] - 6.5) / 2.0 +
        0.4 * df['years_with_diabetes'] / 10.0 +
        0.2 * df['hypertension'] +
        0.1 * (df['systolic_bp'] - 120) / 20.0 -
        0.15 * df['retinal_screening_regularity'] +
        np.random.normal(0, 0.2, n_samples)
    )
    
    # Apply sigmoid to get probability, then convert to binary
    risk_prob = 1 / (1 + np.exp(-risk_score))
    df['referable_DR_risk'] = (risk_prob > 0.35).astype(int)
    
    # Adjust prevalence to ~20% (realistic for diabetic population)
    positive_samples = df['referable_DR_risk'].sum()
    target_prevalence = 0.20
    if positive_samples / n_samples > target_prevalence:
        threshold = np.percentile(risk_prob, 100 * (1 - target_prevalence))
        df['referable_DR_risk'] = (risk_prob > threshold).astype(int)
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_dr_data()
    
    # Save to CSV
    df.to_csv('data/diabetic_retinopathy_data.csv', index=False)
    
    # Print dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['referable_DR_risk'].value_counts(normalize=True))
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset saved to 'data/diabetic_retinopathy_data.csv'")