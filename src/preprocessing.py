from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Import joblib to save the scaler

def preprocess_heart_disease_data():
    # Fetch the dataset
    heart_disease = fetch_ucirepo(id=45)
    
    # Create DataFrames for features and targets
    X = pd.DataFrame(heart_disease.data.features)
    y = pd.DataFrame(heart_disease.data.targets)
    
    # Merge X and y into a single DataFrame
    df = pd.concat([X, y], axis=1)
    
    # Drop all rows with null values
    df = df.dropna()
    
    # Separate features and target again after dropping null values
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Convert target to binary: 0 (no disease), 1 (disease)
    y = y.apply(lambda x: 0 if x == 0 else 1)
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler to a file
    scaler_filename = "scaler.joblib"
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")
    
    # Convert scaled features back to DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Example usage
X_train, X_test, y_train, y_test = preprocess_heart_disease_data()