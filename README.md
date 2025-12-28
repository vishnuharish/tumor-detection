# Breast Cancer Classification

A machine learning project using Gaussian Naive Bayes to classify tumors as malignant or benign, with a FastAPI REST API for easy predictions.

## Overview

This project implements a classification model trained on the breast cancer dataset from Kaggle to predict whether a tumor is malignant or benign. It includes:
- Comprehensive exploratory data analysis and visualization
- Scikit-learn pipeline with StandardScaler preprocessing and GaussianNB classifier
- REST API built with FastAPI for single and batch predictions
- Interactive API documentation with Swagger UI

## Features

- **Exploratory Data Analysis**: Correlation heatmaps, distribution plots, and clustering visualizations
- **Data Preprocessing**: Feature scaling, target encoding, and train-test split
- **Model Pipeline**: Scikit-learn pipeline with StandardScaler and GaussianNB
- **Model Export**: Trained model saved as pickle file for inference
- **FastAPI REST API**: 
  - Single prediction endpoint
  - Batch predictions via CSV file upload
  - Interactive Swagger UI documentation
  - CORS support for cross-origin requests

## Requirements

```
fastapi>=0.128.0
uvicorn[standard]>=0.27.0
pandas>=2.3.3
numpy>=2.4.0
matplotlib>=3.10.8
seaborn>=0.13.2
scikit-learn>=1.8.0
jupyterlab>=4.5.1
pydantic>=2.0.0
```

## Project Structure

```
├── data/
│   ├── tumor.csv
│   └── test.csv
├── models/
│   └── breast_cancer_model.pkl
├── breast-cancer-classification.ipynb
├── main.py
├── model.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation & Setup

1. **Clone/Navigate to project directory**
   ```bash
   cd breast-cancer-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API server**
   ```bash
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /health` - Health status of the API
- `GET /` - Root endpoint with API info

### Predictions
- **Single Prediction**: `POST /predict`
  - Input: JSON with 9 tumor feature values (1-10 scale)
  - Output: Prediction (Benign/Malignant), confidence, and probability scores

  Example request:
  ```json
  {
    "clump_thickness": 5,
    "uniformity_cell_size": 4,
    "uniformity_cell_shape": 4,
    "marginal_adhesion": 5,
    "single_epithelial_cell_size": 7,
    "bare_nuclei": 10,
    "bland_chromatin": 3,
    "normal_nucleoli": 2,
    "mitoses": 1
  }
  ```

- **Batch Predictions**: `POST /predict-batch`
  - Input: CSV file with rows containing the 9 features
  - Output: Predictions for all records in the file
  - CSV columns must include:
    - Clump Thickness
    - Uniformity of Cell Size
    - Uniformity of Cell Shape
    - Marginal Adhesion
    - Single Epithelial Cell Size
    - Bare Nuclei
    - Bland Chromatin
    - Normal Nucleoli
    - Mitoses

### Information
- `GET /info` - Model and API information
- `GET /features` - List of feature names used by the model

## Interactive Documentation

After starting the server, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Model Details

**Algorithm**: Gaussian Naive Bayes with StandardScaler preprocessing

**Classes**:
- Benign (0): Non-cancerous tumors
- Malignant (4): Cancerous tumors

**Features** (9 total):
1. Clump Thickness
2. Uniformity of Cell Size
3. Uniformity of Cell Shape
4. Marginal Adhesion
5. Single Epithelial Cell Size
6. Bare Nuclei
7. Bland Chromatin
8. Normal Nucleoli
9. Mitoses

**Output**:
- Prediction: Class label (Benign or Malignant)
- Confidence: Percentage confidence in the prediction
- Probabilities: Individual probabilities for each class

## Model Performance

The Gaussian Naive Bayes model achieves high accuracy on the test set with detailed classification metrics including precision, recall, and F1-score.

## Data Source

Dataset sourced from Kaggle's breast cancer classification dataset.

## Usage Examples

### Python with requests
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "clump_thickness": 5,
        "uniformity_cell_size": 4,
        "uniformity_cell_shape": 4,
        "marginal_adhesion": 5,
        "single_epithelial_cell_size": 7,
        "bare_nuclei": 10,
        "bland_chromatin": 3,
        "normal_nucleoli": 2,
        "mitoses": 1
    }
)
print(response.json())

# Batch predictions from CSV
with open("test_data.csv", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict-batch", files=files)
    print(response.json())
```

### cURL
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "clump_thickness": 5,
    "uniformity_cell_size": 4,
    "uniformity_cell_shape": 4,
    "marginal_adhesion": 5,
    "single_epithelial_cell_size": 7,
    "bare_nuclei": 10,
    "bland_chromatin": 3,
    "normal_nucleoli": 2,
    "mitoses": 1
  }'

# Batch predictions
curl -X POST "http://localhost:8000/predict-batch" \
  -F "file=@test_data.csv"
```
