"""
FastAPI application for Resume Classification using BERT model.
Provides endpoints for single and batch resume classification.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Analyser API",
    description="API for classifying resumes into 25 job categories using BERT (95.85% accuracy)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job categories
CATEGORIES = [
    "Data Science", "Java Developer", "Testing", "DevOps Engineer",
    "Python Developer", "Web Developer", "HR", "Hadoop", "Blockchain",
    "ETL Developer", "Operations Manager", "Sales", "Mechanical Engineer",
    "Arts", "Database", "Electrical Engineering", "Health and Fitness",
    "PMO", "Business Analyst", "DotNet Developer", "Automation Testing",
    "Network Security Engineer", "SAP Developer", "Civil Engineer", "Advocate"
]

# Global model and tokenizer
model = None
tokenizer = None
device = None

# Pydantic models
class ResumeInput(BaseModel):
    text: str = Field(..., description="Resume text content", min_length=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Skills: Python, Machine Learning, Deep Learning, PyTorch, TensorFlow, NLP. Experience: 3 years in Data Science and AI model development. Projects: Built recommendation systems, sentiment analysis models. Education: Masters in Computer Science"
            }
        }

class BatchResumeInput(BaseModel):
    resumes: List[str] = Field(..., description="List of resume texts", min_items=1, max_items=50)
    
    class Config:
        json_schema_extra = {
            "example": {
                "resumes": [
                    "Python developer with 5 years experience in Django and Flask...",
                    "Experienced data scientist with expertise in machine learning..."
                ]
            }
        }

class PredictionOutput(BaseModel):
    category: str = Field(..., description="Predicted job category")
    category_id: int = Field(..., description="Category ID (0-24)")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    all_scores: Optional[dict] = Field(None, description="Scores for all categories")
    
class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_processed: int
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    categories_count: int
    timestamp: str

@app.on_event("startup")
async def load_model():
    """Load the model and tokenizer on startup."""
    global model, tokenizer, device
    
    try:
        logger.info("Loading model and tokenizer...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Try to load from local path first, then from Hugging Face Hub
        import os
        local_model_path = "../models/saved_models/resume_analyser_bert"
        
        if os.path.exists(local_model_path):
            logger.info(f"Loading model from local path: {local_model_path}")
            model = BertForSequenceClassification.from_pretrained(local_model_path)
            tokenizer = BertTokenizer.from_pretrained(local_model_path)
        else:
            logger.info("Loading model from Hugging Face Hub...")
            model = BertForSequenceClassification.from_pretrained('SwaKyxd/resume-analyser-bert')
            tokenizer = BertTokenizer.from_pretrained('SwaKyxd/resume-analyser-bert')
        
        # Move model to device and set to eval mode
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Resume Analyser API - Enhanced Model (95.85% Accuracy)",
        "version": "2.0.0",
        "model_config": "config_6_moderate",
        "accuracy": "95.85%",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "categories": "/categories",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=str(device),
        categories_count=len(CATEGORIES),
        timestamp=datetime.now().isoformat()
    )

@app.get("/categories", response_model=dict)
async def get_categories():
    """Get all available job categories."""
    return {
        "total": len(CATEGORIES),
        "categories": {i: cat for i, cat in enumerate(CATEGORIES)}
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(resume: ResumeInput, include_all_scores: bool = False):
    """
    Classify a single resume into a job category.
    
    Args:
        resume: Resume text input
        include_all_scores: Whether to include scores for all categories
    
    Returns:
        PredictionOutput with category, confidence, and optionally all scores
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize
        inputs = tokenizer(
            resume.text,
            return_tensors='pt',
            max_length=200,
            padding='max_length',
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction[0]].item()
        
        # Prepare response
        category_id = prediction.item()
        category = CATEGORIES[category_id]
        
        result = PredictionOutput(
            category=category,
            category_id=category_id,
            confidence=confidence,
            all_scores=None
        )
        
        # Add all scores if requested
        if include_all_scores:
            all_scores = {
                CATEGORIES[i]: float(probabilities[0][i].item())
                for i in range(len(CATEGORIES))
            }
            result.all_scores = all_scores
        
        logger.info(f"Prediction: {category} (confidence: {confidence:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionOutput)
async def batch_predict(batch: BatchResumeInput, include_all_scores: bool = False):
    """
    Classify multiple resumes in a batch.
    
    Args:
        batch: List of resume texts
        include_all_scores: Whether to include scores for all categories
    
    Returns:
        BatchPredictionOutput with predictions for all resumes
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize batch
        inputs = tokenizer(
            batch.resumes,
            return_tensors='pt',
            max_length=200,
            padding='max_length',
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Prepare results
        results = []
        for i, pred in enumerate(predictions):
            category_id = pred.item()
            category = CATEGORIES[category_id]
            confidence = probabilities[i][pred].item()
            
            result = PredictionOutput(
                category=category,
                category_id=category_id,
                confidence=confidence,
                all_scores=None
            )
            
            # Add all scores if requested
            if include_all_scores:
                all_scores = {
                    CATEGORIES[j]: float(probabilities[i][j].item())
                    for j in range(len(CATEGORIES))
                }
                result.all_scores = all_scores
            
            results.append(result)
        
        logger.info(f"Batch prediction completed: {len(results)} resumes processed")
        
        return BatchPredictionOutput(
            predictions=results,
            total_processed=len(results)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
