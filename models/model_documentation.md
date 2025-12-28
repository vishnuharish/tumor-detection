# Model Documentation

## Overview

This document provides comprehensive documentation for the Breast Cancer Classification model used in this project.

## Model Specification

### Algorithm
- **Name**: Gaussian Naive Bayes (GaussianNB)
- **Type**: Probabilistic Classification Algorithm
- **Framework**: Scikit-learn
- **Preprocessing**: StandardScaler (zero mean, unit variance)

### Model Pipeline

The model uses a scikit-learn Pipeline that combines preprocessing and classification:

```
Pipeline Steps:
1. StandardScaler - Normalize feature values to have mean 0 and std 1
2. GaussianNB - Gaussian Naive Bayes classifier
```

## Input Features

### Feature Details

The model accepts **9 input features**, all on a scale of 1-10:

| Feature # | Feature Name | Scale | Description |
|-----------|--------------|-------|-------------|
| 1 | Clump Thickness | 1-10 | Thickness of cancer cell clumps |
| 2 | Uniformity of Cell Size | 1-10 | Uniformity in size of cells within the tumor |
| 3 | Uniformity of Cell Shape | 1-10 | Uniformity in shape of cells within the tumor |
| 4 | Marginal Adhesion | 1-10 | Degree of adhesion between cell and margin |
| 5 | Single Epithelial Cell Size | 1-10 | Size of single epithelial cells |
| 6 | Bare Nuclei | 1-10 | Proportion of cells with bare nuclei |
| 7 | Bland Chromatin | 1-10 | Texture of chromatin in nuclei |
| 8 | Normal Nucleoli | 1-10 | Prominence of nucleoli |
| 9 | Mitoses | 1-10 | Number of mitoses |

### Input Validation

All input features must satisfy:
- Type: Numeric (integer or float)
- Range: 1.0 to 10.0 (inclusive)
- Missing values: Not allowed

## Output Classes

The model performs **binary classification** with two classes:

### Class Definitions

| Class | Label | Value | Description |
|-------|-------|-------|-------------|
| Benign | 2 | 0 | Non-cancerous tumor |
| Malignant | 4 | 1 | Cancerous tumor |

### Output Format

Each prediction includes:

```json
{
  "prediction": "Benign",
  "confidence": "95.23%",
  "benign_probability": 95.23,
  "malignant_probability": 4.77
}
```

## Model Architecture

### Gaussian Naive Bayes

**How it Works:**

The Gaussian Naive Bayes classifier is based on Bayes' theorem and assumes that features follow a Gaussian (normal) distribution:

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

Where:
- $P(y|X)$ = Posterior probability (what we want to predict)
- $P(X|y)$ = Likelihood of features given class
- $P(y)$ = Prior probability of class
- $P(X)$ = Evidence

**Advantages:**
- Fast training and prediction
- Works well with small datasets
- Provides probability estimates
- Interpretable results
- Low computational complexity

**Limitations:**
- Assumes feature independence (not always true)
- Assumes Gaussian distribution of features
- May underperform on complex relationships

## Training Process

### Data Preparation

```
Total Dataset: [number of samples]
├── Training Set: 80% (used to train the model)
└── Test Set: 20% (used to evaluate the model)
```

### Training Details

- **Random State**: 42 (for reproducibility)
- **Test Size**: 0.2 (20% of data)
- **Scaler**: StandardScaler (fit on training data)
- **Feature Scaling**: Applied before classification

### Training Steps

1. Load breast cancer dataset from CSV
2. Extract features (9 columns) and target (Class column)
3. Split data into training (80%) and test (20%) sets
4. Fit StandardScaler on training features
5. Train GaussianNB on scaled training data
6. Evaluate on test data
7. Save pipeline to pickle file

## Model Performance

### Metrics

The model is evaluated using standard classification metrics:

- **Accuracy**: Percentage of correct predictions
- **Precision**: Correct positive predictions / all positive predictions
- **Recall**: Correct positive predictions / all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve

### Performance Evaluation

Classification metrics are calculated for:
- **Training Set**: Performance on data used to train
- **Test Set**: Performance on unseen data (generalization)

### Confusion Matrix

```
                Predicted
               Benign  Malignant
Actual Benign    TN      FP
       Malignant FN      TP
```

Where:
- **TP (True Positive)**: Correctly predicted Malignant
- **TN (True Negative)**: Correctly predicted Benign
- **FP (False Positive)**: Benign predicted as Malignant (Type I Error)
- **FN (False Negative)**: Malignant predicted as Benign (Type II Error - more critical)

## Model Persistence

### Serialization

The trained model is saved as a Python pickle file:

```
Location: models/breast_cancer_model.pkl
Size: ~[size in KB]
Format: Binary pickle format (Python's serialization protocol)
```

### Model Loading

The model is automatically loaded from disk when making predictions:

```python
import pickle

with open('models/breast_cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## Data Source

### Dataset Information

- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: Kaggle
- **Samples**: [number of tumor samples]
- **Features**: 9 diagnostic measurements
- **Classes**: 2 (Benign and Malignant)

### CSV Files

- **Training Data**: `data/tumor.csv`
- **Test Data**: `data/test.csv`

### Data Characteristics

- **Class Distribution**: [distribution info]
- **Missing Values**: None (clean dataset)
- **Outliers**: Minimal

## Assumptions & Limitations

### Model Assumptions

1. **Feature Independence**: Features are assumed to be independent (Naive Bayes assumption)
2. **Gaussian Distribution**: Each feature follows a normal distribution within each class
3. **Balanced Features**: All features are equally important
4. **Fixed Ranges**: Features are on 1-10 scale in all data

### Limitations

1. **Small Dataset**: May not capture all complex patterns
2. **Linearity**: Assumes linear relationships in feature space
3. **Class Imbalance**: If classes are imbalanced, accuracy may be misleading
4. **Medical Decision**: Not suitable as sole decision for medical diagnosis
5. **Feature Engineering**: May benefit from additional engineered features

## Usage Guidelines

### When to Use

- Quick baseline model for tumor classification
- When interpretability is important
- For datasets with limited samples
- When computational efficiency is needed
- For probability-based predictions

### When NOT to Use

- As sole medical decision-making tool
- With features outside 1-10 range
- When missing values are present
- For real-time critical applications without validation
- With completely different feature sets

## Retraining the Model

### Steps to Retrain

```python
from model import train_model

# Retrain the model with latest data
new_model = train_model()
```

### When to Retrain

- After collecting significant new data
- When model performance degrades
- When data distribution changes
- Periodic retraining (e.g., quarterly)
- After fixing data quality issues

## API Integration

### Prediction Endpoints

#### Single Prediction
```
POST /predict
```

Accepts: JSON with 9 feature values
Returns: Prediction with confidence and probabilities

#### Batch Prediction
```
POST /predict-batch
```

Accepts: CSV file with multiple records
Returns: Array of predictions

#### Model Information
```
GET /info
```

Returns: Model type, algorithm, features, classes

## Monitoring & Maintenance

### Key Metrics to Monitor

- Prediction accuracy on new data
- Class distribution in predictions
- Feature value ranges in incoming requests
- API response times
- Error rates

### Maintenance Tasks

1. **Regular Retraining**: Retrain model with new data
2. **Performance Monitoring**: Track prediction accuracy
3. **Data Quality**: Monitor incoming data for anomalies
4. **Version Control**: Keep track of model versions
5. **Documentation**: Update docs with new findings

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-28 | Initial model release with FastAPI integration |

## References

- [Gaussian Naive Bayes - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [Breast Cancer Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- [Naive Bayes Classifier Theory](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

## Contact & Support

For questions or issues related to the model, please refer to the main README.md file or contact the development team.
