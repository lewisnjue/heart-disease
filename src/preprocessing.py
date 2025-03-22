from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_heart_disease_data():
    heart_disease = fetch_ucirepo(id=45)
    
    X = pd.DataFrame(heart_disease.data.features)
    y = pd.DataFrame(heart_disease.data.targets)
    
    df = pd.concat([X, y], axis=1)
    
    df = df.dropna()
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    y = y.apply(lambda x: 0 if x == 0 else 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
