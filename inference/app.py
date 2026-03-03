"""
ML Service - T1: Disaster Relevance | T2: Help Request & Humanitarian Category
FastAPI servisi - Reddit post analizi (T1: afet ilgisi, T2: yardım talebi + kategori)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict
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
from t2_humanitarian import get_t2_classifier

# FastAPI app oluştur
app = FastAPI(
    title="ML Service - T1 & T2 Text Analysis",
    description="T1: Disaster Relevance | T2: Help Request & Humanitarian Category Detection",
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
    use_roberta: Optional[bool] = Field(False, description="True ise RoBERTa modeli kullanılır (model mevcutsa)")


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
    model_used: str = Field(..., description="Kullanılan model: 'logistic_regression' veya 'roberta'")


# --- Birleşik T1+T2 (tek istek) ---
class UnifiedAnalysisRequest(BaseModel):
    """Tek istekte T1 + T2 analizi"""
    text: str = Field(..., description="Analiz edilecek metin", min_length=1)
    use_roberta: Optional[bool] = Field(False, description="True ise RoBERTa (T1)")
    include_t2: Optional[bool] = Field(True, description="True ise aynı istekte T2 (yardım talebi + kategori) de döner")


class T2Result(BaseModel):
    """T2 çıktısı (birleşik response içinde)"""
    is_help_request: bool
    help_request_probability: float
    humanitarian_labels: List[str]
    category_probabilities: Dict[str, float]


class UnifiedAnalysisResponse(BaseModel):
    """T1 + T2 tek response"""
    is_disaster_related: bool
    relevance_score: float
    message: str
    model_used: str
    t2: Optional[T2Result] = Field(None, description="T2 sonucu (include_t2=True ise dolu)")


# --- T2: Help Request & Humanitarian Category ---
class T2AnalysisRequest(BaseModel):
    """T2 analizi için request (disaster-related post metni)"""
    text: str = Field(..., description="Afet ile ilgili post metni", min_length=1)


class T2AnalysisResponse(BaseModel):
    """T2 response: yardım talebi + insani kategori(ler) ve olasılıklar"""
    is_help_request: bool = Field(..., description="Yardım talebi mi?")
    help_request_probability: float = Field(..., description="Yardım talebi olasılığı 0-1")
    humanitarian_labels: List[str] = Field(
        ...,
        description="İnsani kategori etiketleri: urgent_needs, infrastructure_damage, donations_volunteering, other"
    )
    category_probabilities: Dict[str, float] = Field(
        ...,
        description="Her kategori için olasılık (toplamı 1)"
    )


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
        # Text'i analiz et (Logistic Regression veya RoBERTa + Translate-and-Test)
        is_related, score, model_used = text_analyzer.classify_disaster_relevance(
            request.text, use_roberta=request.use_roberta or False
        )
        
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
            model_used=model_used
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text analizi sırasında hata oluştu: {str(e)}"
        )


@app.post("/analyze", response_model=UnifiedAnalysisResponse)
async def analyze_unified(request: UnifiedAnalysisRequest):
    """
    Tek istekte T1 + T2. use_roberta=True ve multi-task model varsa T1+T2 tek modelden döner.
    """
    try:
        is_related, score, model_used, t2_from_model = text_analyzer.classify_disaster_relevance(
            request.text, use_roberta=request.use_roberta or False, return_t2=request.include_t2
        )
        percentage = score * 100
        message = f"Post afet ile ilgili (güven: {percentage:.2f}%)" if is_related else f"Post afet ile ilgili değil (güven: {percentage:.2f}%)"

        t2_result = None
        if request.include_t2:
            if t2_from_model is not None:
                t2_result = T2Result(
                    is_help_request=t2_from_model["is_help_request"],
                    help_request_probability=t2_from_model["help_request_probability"],
                    humanitarian_labels=t2_from_model["humanitarian_labels"],
                    category_probabilities=t2_from_model["category_probabilities"],
                )
            else:
                t2 = get_t2_classifier().analyze(request.text)
                t2_result = T2Result(
                    is_help_request=t2["is_help_request"],
                    help_request_probability=t2["help_request_probability"],
                    humanitarian_labels=t2["humanitarian_labels"],
                    category_probabilities=t2["category_probabilities"],
                )

        return UnifiedAnalysisResponse(
            is_disaster_related=is_related,
            relevance_score=round(score, 2),
            message=message,
            model_used=model_used,
            t2=t2_result,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


@app.post("/t2/analyze", response_model=T2AnalysisResponse)
async def analyze_t2(request: T2AnalysisRequest):
    """
    T2: Help Request and Humanitarian Category Detection
    Disaster-related post için: yardım talebi mi, hangi insani kategori(ler).
    Çıktı: discrete label(s) + class probabilities.
    """
    try:
        classifier = get_t2_classifier()
        result = classifier.analyze(request.text)
        return T2AnalysisResponse(
            is_help_request=result["is_help_request"],
            help_request_probability=result["help_request_probability"],
            humanitarian_labels=result["humanitarian_labels"],
            category_probabilities=result["category_probabilities"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"T2 analizi sırasında hata: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint - servis bilgisi"""
    return {
        "service": "ml-service",
        "tasks": ["T1: Disaster Relevance", "T2: Help Request & Humanitarian Category"],
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (T1+T2 tek istek)",
            "t1_analyze": "/t1/analyze",
            "t2_analyze": "/t2/analyze",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
