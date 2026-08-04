"""FastAPI приложение для поиска аналогов ЛД."""

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

from search_engine import SearchEngine, parse_query
from config import DEFAULT_MODEL_NAME


app = FastAPI(
    title="LD Analog Search API",
    description="Гибридный поиск (Dense + BM25) аналогов ЛД через Qdrant",
    version="1.0.0",
)

# Глобальный инстанс движка (можно заменить на dependency injection)
engine = SearchEngine(model_name=DEFAULT_MODEL_NAME)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    use_hybrid: bool = True


class SearchResponse(BaseModel):
    query: str
    parsed: Dict[str, Any]
    count: int
    results: List[Dict[str, Any]]


@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Универсальный поиск по названию, артикулу, бренду или характеристикам.
    
    Примеры запросов:
    - "Мне нужен аналог крана Temper Ду80 Ру16"
    - "ALSO Ду200 Ру40"
    - "артикул 12345-AB"
    """
    parsed = parse_query(request.query)
    results = engine.search(
        request.query,
        top_k=request.top_k,
        use_hybrid=request.use_hybrid,
    )
    
    return SearchResponse(
        query=request.query,
        parsed={
            "brand": parsed.brand,
            "article": parsed.article,
            "equipment_type": parsed.equipment_type,
            "dn": parsed.dn,
            "pn": parsed.pn,
            "is_article_query": parsed.is_article_query,
        },
        count=len(results),
        results=results,
    )


@app.post("/analogs", response_model=SearchResponse)
async def find_analogs(request: SearchRequest):
    """
    Ищет аналоги ЛД для заданного запроса.
    
    Сначала находит товар, затем:
    - Если у него есть поле ld_analog — ищет по нему
    - Иначе возвращает семантически похожие товары
    """
    parsed = parse_query(request.query)
    results = engine.find_analogs(request.query, top_k=request.top_k)
    
    return SearchResponse(
        query=request.query,
        parsed={
            "brand": parsed.brand,
            "article": parsed.article,
            "equipment_type": parsed.equipment_type,
            "dn": parsed.dn,
            "pn": parsed.pn,
        },
        count=len(results),
        results=results,
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": engine.model_name}


@app.get("/compare-models")
async def compare_models(
    query: str = Query(..., description="Тестовый запрос"),
    models: str = Query(
        "all-MiniLM-L6-v2,paraphrase-multilingual-MiniLM-L12-v2",
        description="Список моделей через запятую"
    ),
    top_k: int = 5,
):
    """Сравнивает результаты разных embedding-моделей на одном запросе."""
    model_list = [m.strip() for m in models.split(",")]
    comparison = engine.compare_models(query, model_list, top_k=top_k)
    return {"query": query, "comparison": comparison}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)