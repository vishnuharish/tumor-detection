import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

# Feature names for the model
FEATURE_NAMES = [
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses'
]

MODEL_PATH = 'models/breast_cancer_model.pkl'
SCALER_PATH = 'models/scaler.pkl'


def train_model():
    """Train and save the breast cancer classification model"""
    # Load data
    df = pd.read_csv('./data/tumor.csv')
    
    # Prepare features and target
    X = df.drop(columns=['Sample code number', 'Class'])
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    return pipeline


def load_model():
    """Load the trained model from disk"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("Model not found. Training new model...")
        return train_model()


def predict(features: list) -> dict:
    """
    Make a prediction on the given features
    
    Args:
        features: List of 9 feature values
        
    Returns:
        Dictionary with prediction and confidence
    """
    model = load_model()
    
    # Convert to numpy array and reshape
    X = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Map prediction to class name
    class_name = "Malignant" if prediction == 4 else "Benign"
    confidence = float(max(probabilities) * 100)
    
    return {
        "prediction": class_name,
        "confidence": f"{confidence:.2f}%",
        "benign_probability": float(probabilities[0] * 100),
        "malignant_probability": float(probabilities[1] * 100)
    }
