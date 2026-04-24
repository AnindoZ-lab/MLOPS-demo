import json
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import time

def train():
    # Create models directory if it doesn't exist
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names.tolist()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    
    # Compile metrics dictionary
    metrics = {
        "model_info": {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "training_time_seconds": round(training_time, 4),
            "dataset": "Iris",
            "num_classes": len(target_names),
            "classes": target_names,
            "num_features": len(feature_names),
            "features": list(feature_names),
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0])
        },
        "performance_metrics": {
            "accuracy": round(accuracy, 4),
            "precision_weighted": round(precision, 4),
            "recall_weighted": round(recall, 4),
            "f1_score_weighted": round(f1, 4),
            "confusion_matrix": conf_matrix,
            "cross_val_scores": {
                "mean": round(cv_scores.mean(), 4),
                "std": round(cv_scores.std(), 4),
                "all_scores": [round(score, 4) for score in cv_scores]
            }
        },
        "feature_importance": feature_importance,
        "class_distribution": {
            "train": dict(zip(*np.unique(y_train, return_counts=True))),
            "test": dict(zip(*np.unique(y_test, return_counts=True))),
            "full_dataset": dict(zip(*np.unique(y, return_counts=True)))
        }
    }
    
    # Save model
    model_path = models_dir / "model.joblib"
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save metrics
    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: Random Forest Classifier")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"\nCross-validation (5-fold): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"\nTraining time: {training_time:.3f} seconds")
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    print("="*50)
    
    return model, metrics

def load_and_inspect_model():
    """Helper function to verify saved model works"""
    models_dir = Path("../models")
    model_path = models_dir / "model.joblib"
    metrics_path = models_dir / "metrics.json"
    
    if model_path.exists():
        loaded_model = joblib.load(model_path)
        print(f"\n✓ Model loaded successfully from {model_path}")
        
        # Quick test prediction
        test_sample = [[5.1, 3.5, 1.4, 0.2]]  # Sample from iris
        prediction = loaded_model.predict(test_sample)
        probability = loaded_model.predict_proba(test_sample)
        print(f"Test prediction: {prediction[0]}")
        print(f"Prediction probability: {probability[0]}")
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            loaded_metrics = json.load(f)
        print(f"\n✓ Metrics loaded successfully from {metrics_path}")
        print(f"  Model accuracy from metrics: {loaded_metrics['performance_metrics']['accuracy']}")

if __name__ == "__main__":
    # Train model and save artifacts
    model, metrics = train()
    
    # Verify saved artifacts work
    load_and_inspect_model()
