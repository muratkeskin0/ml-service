"""
ML Service - T1: Disaster Relevance Classification
FastAPI servisi - Reddit post'larının afet ile ilgili olup olmadığını analiz eder
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# TextAnalyzer instance oluştur (Logistic Regression + Ücretsiz Multi-Language Translation)
print("[INFO] Logistic Regression modeli yükleniyor")
print("[INFO] Ücretsiz multi-language translation aktif (100+ dil → İngilizce)")
text_analyzer = TextAnalyzer(enable_translation=True)


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
        # Text'i analiz et (Logistic Regression + Translate-and-Test)
        is_related, score = text_analyzer.classify_disaster_relevance(request.text)
        
        # Message oluştur
        percentage = score * 100
        if is_related:
            message = f"Post afet ile ilgili (güven: {percentage:.2f}%)"
        else:
            message = f"Post afet ile ilgili değil (güven: {percentage:.2f}%)"
        
        return TextAnalysisResponse(
            is_disaster_related=is_related,
            relevance_score=round(score, 2),
            message=message
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
