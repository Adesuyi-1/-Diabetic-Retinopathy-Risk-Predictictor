import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data(test_size=0.2, random_state=42, apply_smote=True):
    """
    Preprocess the diabetic retinopathy dataset.
    Returns preprocessed train/test splits and saves preprocessing objects.
    """
    
    # Load data
    df = pd.read_csv('data/diabetic_retinopathy_data.csv')
    
    # Separate features and target
    X = df.drop('referable_DR_risk', axis=1)
    y = df['referable_DR_risk']
    
    # Define feature types
    numerical_features = ['age', 'bmi', 'hbA1c', 'systolic_bp', 'diastolic_bp', 
                         'years_with_diabetes', 'serum_cholesterol']
    
    categorical_features = ['gender', 'smoking_status', 'hypertension', 
                           'dyslipidemia', 'family_history_dr', 'retinal_screening_regularity']
    
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data BEFORE applying SMOTE (to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Original train shape: {X_train.shape}")
    print(f"Original class distribution in train:")
    print(y_train.value_counts(normalize=True))
    
    # Apply SMOTE only to training data
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"\nAfter SMOTE train shape: {X_train_resampled.shape}")
        print(f"Resampled class distribution in train:")
        print(pd.Series(y_train_resampled).value_counts(normalize=True))
        
        # Fit preprocessor on resampled training data
        X_train_processed = preprocessor.fit_transform(X_train_resampled)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        numerical_features_processed = preprocessor.named_transformers_['num'].get_feature_names_out(numerical_features)
        categorical_features_processed = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = list(numerical_features_processed) + list(categorical_features_processed)
        
        # Save preprocessing objects
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        joblib.dump(smote, 'models/smote.pkl')
        
        return (X_train_processed, X_test_processed, 
                y_train_resampled, y_test, all_features, preprocessor)
    
    else:
        # Without SMOTE
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names
        numerical_features_processed = preprocessor.named_transformers_['num'].get_feature_names_out(numerical_features)
        categorical_features_processed = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = list(numerical_features_processed) + list(categorical_features_processed)
        
        # Save preprocessor
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        
        return (X_train_processed, X_test_processed, 
                y_train, y_test, all_features, preprocessor)

def get_preprocessed_dataframes():
    """
    Return processed data as DataFrames with proper column names.
    Useful for SHAP analysis and interpretation.
    """
    X_train, X_test, y_train, y_test, feature_names, preprocessor = preprocess_data()
    
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['referable_DR_risk'] = y_train.values
    
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['referable_DR_risk'] = y_test.values
    
    return train_df, test_df, feature_names

if __name__ == "__main__":
    print("=== Data Preprocessing Pipeline ===\n")
    
    # Run preprocessing
    X_train, X_test, y_train, y_test, feature_names, preprocessor = preprocess_data()
    
    print(f"\nPreprocessing completed!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    
    # Save processed data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    # Save feature names
    with open('data/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"\nProcessed data saved to data/ directory")