"""
ML Service - T1: Disaster Relevance Classification
FastAPI servisi - Reddit post'larının afet ile ilgili olup olmadığını analiz eder
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os
import codecs

# Windows için stdout/stderr UTF-8 ayarla (Turkce karakter hatalarını önlemek için)
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# services klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

from text_analyzer.text_analyzer import TextAnalyzer
from text_analyzer.ensemble_analyzer import EnsembleTextAnalyzer

# FastAPI app oluştur
app = FastAPI(
    title="ML Service - T1 Disaster Analysis",
    description="T1: Disaster Relevance Classification Service",
    version="1.0.0"
)

# CORS middleware ekle (Spring Boot'tan çağrılabilmesi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da specific origins belirtilmeli
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model seçimi: ensemble veya tek model
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "true").lower() == "true"
LOGISTIC_WEIGHT = float(os.getenv("LOGISTIC_WEIGHT", "0.3"))
XLM_ROBERTA_WEIGHT = float(os.getenv("XLM_ROBERTA_WEIGHT", "0.7"))

# TextAnalyzer instance oluştur
if USE_ENSEMBLE:
    print("[INFO] Ensemble mode aktif - Her iki model birlikte kullanılacak")
    text_analyzer = EnsembleTextAnalyzer(
        logistic_weight=LOGISTIC_WEIGHT,
        xlm_roberta_weight=XLM_ROBERTA_WEIGHT
    )
else:
    print("[INFO] Single model mode - Otomatik model seçimi")
    text_analyzer = TextAnalyzer()


# Request/Response Models
class TextAnalysisRequest(BaseModel):
    """Text analizi için request model"""
    text: str = Field(..., description="Analiz edilecek metin", min_length=1)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    service: str
    task: str


class TextAnalysisResponse(BaseModel):
    """Text analizi response model"""
    is_disaster_related: bool
    relevance_score: float
    message: str
    ensemble_details: Optional[dict] = None  # Ensemble kullanılıyorsa detaylar


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Servisin çalışıp çalışmadığını kontrol eder
    """
    return HealthResponse(
        status="ok",
        service="ml-service-t1",
        task="Disaster Relevance Classification"
    )


@app.post("/t1/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    T1: Disaster Relevance Classification
    Reddit post'unun afet ile ilgili olup olmadığını analiz eder
    
    Args:
        request: TextAnalysisRequest - Analiz edilecek metin
        
    Returns:
        TextAnalysisResponse - is_disaster_related, relevance_score, message
    """
    try:
        # Text'i analiz et
        if USE_ENSEMBLE and isinstance(text_analyzer, EnsembleTextAnalyzer):
            # Ensemble mode
            is_related, score, details = text_analyzer.classify_disaster_relevance(request.text)
            ensemble_details = details
        else:
            # Single model mode
            is_related, score = text_analyzer.classify_disaster_relevance(request.text)
            ensemble_details = None
        
        # Message oluştur
        percentage = score * 100
        if is_related:
            message = f"Post afet ile ilgili (güven: {percentage:.2f}%)"
        else:
            message = f"Post afet ile ilgili değil (güven: {percentage:.2f}%)"
        
        return TextAnalysisResponse(
            is_disaster_related=is_related,
            relevance_score=round(score, 2),
            message=message,
            ensemble_details=ensemble_details
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text analizi sırasında hata oluştu: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint - servis bilgisi"""
    return {
        "service": "ml-service-t1",
        "task": "Disaster Relevance Classification",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/t1/analyze"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
