from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import pandas as pd
import io
from model import predict, FEATURE_NAMES

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Classification API",
    description="API for classifying breast cancer tumors as benign or malignant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    clump_thickness: float = Field(..., ge=1, le=10, description="Clump Thickness (1-10)")
    uniformity_cell_size: float = Field(..., ge=1, le=10, description="Uniformity of Cell Size (1-10)")
    uniformity_cell_shape: float = Field(..., ge=1, le=10, description="Uniformity of Cell Shape (1-10)")
    marginal_adhesion: float = Field(..., ge=1, le=10, description="Marginal Adhesion (1-10)")
    single_epithelial_cell_size: float = Field(..., ge=1, le=10, description="Single Epithelial Cell Size (1-10)")
    bare_nuclei: float = Field(..., ge=1, le=10, description="Bare Nuclei (1-10)")
    bland_chromatin: float = Field(..., ge=1, le=10, description="Bland Chromatin (1-10)")
    normal_nucleoli: float = Field(..., ge=1, le=10, description="Normal Nucleoli (1-10)")
    mitoses: float = Field(..., ge=1, le=10, description="Mitoses (1-10)")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint"""
    prediction: str
    confidence: str
    benign_probability: float
    malignant_probability: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Breast Cancer Classification API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running successfully"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def make_prediction(request: PredictionRequest):
    """
    Make a prediction for breast cancer classification
    
    The model takes 9 features as input:
    - Clump Thickness
    - Uniformity of Cell Size
    - Uniformity of Cell Shape
    - Marginal Adhesion
    - Single Epithelial Cell Size
    - Bare Nuclei
    - Bland Chromatin
    - Normal Nucleoli
    - Mitoses
    
    Returns:
    - prediction: "Benign" or "Malignant"
    - confidence: Confidence percentage of the prediction
    - benign_probability: Probability of benign (0-100)
    - malignant_probability: Probability of malignant (0-100)
    """
    try:
        features = [
            request.clump_thickness,
            request.uniformity_cell_size,
            request.uniformity_cell_shape,
            request.marginal_adhesion,
            request.single_epithelial_cell_size,
            request.bare_nuclei,
            request.bland_chromatin,
            request.normal_nucleoli,
            request.mitoses
        ]
        
        result = predict(features)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@app.get("/features", tags=["Information"])
async def get_features():
    """Get list of feature names expected by the model"""
    return {
        "features": FEATURE_NAMES,
        "count": len(FEATURE_NAMES),
        "description": "Features used for breast cancer classification"
    }


@app.get("/info", tags=["Information"])
async def get_info():
    """Get API information"""
    return {
        "title": "Breast Cancer Classification API",
        "description": "Classifies tumors as benign or malignant using Gaussian Naive Bayes",
        "model": "Gaussian Naive Bayes",
        "features": len(FEATURE_NAMES),
        "algorithm": "Gaussian Naive Bayes with StandardScaler preprocessing",
        "classes": ["Benign", "Malignant"]
    }


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction endpoint that accepts a CSV file
    
    CSV File Requirements:
    - Must contain columns matching the feature names (case-insensitive)
    - Expected columns: Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape,
      Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin,
      Normal Nucleoli, Mitoses
    - All feature values must be between 1-10
    
    Returns:
    - List of predictions with confidence scores for each row
    """
    try:
        # Read CSV file
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="File must be a CSV file"
            )
        
        # Read file content
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Validate columns exist
        required_columns = [
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
        
        # Try to match columns (case-insensitive)
        df_columns_lower = {col.lower(): col for col in df.columns}
        required_columns_lower = {col.lower(): col for col in required_columns}
        
        missing_columns = []
        for req_col_lower in required_columns_lower:
            if req_col_lower not in df_columns_lower:
                missing_columns.append(req_col_lower)
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}. Expected: {required_columns}"
            )
        
        # Extract features in correct order
        feature_data = []
        for req_col in required_columns:
            req_col_lower = req_col.lower()
            actual_col = df_columns_lower[req_col_lower]
            feature_data.append(df[actual_col].values)
        
        # Create feature matrix
        X = pd.DataFrame(feature_data).T
        
        # Validate feature values
        if (X < 1).any().any() or (X > 10).any().any():
            raise HTTPException(
                status_code=400,
                detail="All feature values must be between 1 and 10"
            )
        
        # Make predictions
        predictions = []
        for idx, row in X.iterrows():
            try:
                result = predict(row.values.tolist())
                predictions.append({
                    "row": idx + 1,
                    **result
                })
            except Exception as e:
                predictions.append({
                    "row": idx + 1,
                    "error": str(e)
                })
        
        return {
            "file_name": file.filename,
            "total_records": len(df),
            "predictions": predictions
        }
        
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV parsing error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
