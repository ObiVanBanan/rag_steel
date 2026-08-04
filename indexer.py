"""Индексация CSV в Qdrant с извлечением бренда из названия."""

import pandas as pd
import re
from typing import Optional
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from config import DEFAULT_MODEL_NAME, QDRANT_URL, COLLECTION_NAME, MODEL_REGISTRY


def extract_numeric(value) -> Optional[float]:
    """Извлекает число из строки вида 'Ду80', 'Ру16', '200 мм'."""
    if pd.isna(value):
        return None
    s = str(value)
    nums = re.findall(r"[\d]+[.,]?\d*", s.replace(",", "."))
    return float(nums[0]) if nums else None


def extract_brand(name: str) -> Optional[str]:
    """Извлекает бренд из названия товара."""
    if not isinstance(name, str):
        return None
    name_lower = name.lower()
    known_brands = {
        "temper": "Temper",
        "also": "ALSO",
        "алсо": "ALSO",
        "marshal": "MARSHAL",
        "маршал": "MARSHAL",
        "broen": "Broen",
        "bival": "Бивал",
        "forteca": "FORTECA",
        "ld": "LD",
    }
    for alias, canonical in known_brands.items():
        if alias in name_lower:
            return canonical
    # Если не нашли, пробуем найти слово с заглавной буквы
    words = name.split()
    for w in words:
        if w[0].isupper() and len(w) > 2 and w.lower() not in (
            "кран", "шаровой", "стальной", "из", "на", "для", "с", "под", "полнопроходной"
        ):
            return w
    return None


def build_search_text(row: pd.Series) -> str:
    """Формирует текст для эмбеддинга из структурированных полей."""
    parts = []
    if pd.notna(row.get("name")):
        parts.append(str(row["name"]))
    if pd.notna(row.get("brand")):
        parts.append(str(row["brand"]))
    if pd.notna(row.get("type")):
        parts.append(str(row["type"]))
    if pd.notna(row.get("article")):
        parts.append(str(row["article"]))
    if pd.notna(row.get("dn")):
        parts.append(f"Ду{row['dn']}")
    if pd.notna(row.get("pn")):
        parts.append(f"Ру{row['pn']}")
    if pd.notna(row.get("material")):
        parts.append(str(row["material"]))
    if pd.notna(row.get("connection")):
        parts.append(str(row["connection"]))
    return " ".join(parts)


def index_csv(
    csv_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    limit: Optional[int] = None,
):
    """Индексирует CSV в Qdrant с двумя типами векторов."""
    
    model = MODEL_REGISTRY[model_name]()
    client = QdrantClient(url=QDRANT_URL)
    
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    
    # --- Нормализация полей ---
    column_map = {
        "article": ["article", "vendor_code", "артикул", "код", "sku"],
        "brand": ["brand", "manufacturer", "бренд", "производитель", "торговая_марка"],
        "name": ["name", "наименование", "название", "product_name"],
        "type": ["type", "тип", "product_type"],
        "dn": ["dn", "du", "ду", "diameter", "диаметр"],
        "pn": ["pn", "ru", "ру", "pressure", "давление"],
        "material": ["material", "материал"],
        "connection": ["connection", "присоединение", "соединение"],
        "ld_analog": ["ld_analog", "аналог_лд", "analog", "аналог"],
    }
    
    df_cols_lower = {c.lower(): c for c in df.columns}
    fields = {}
    for field, candidates in column_map.items():
        for cand in candidates:
            if cand.lower() in df_cols_lower:
                fields[field] = df_cols_lower[cand.lower()]
                break
    
    print(f"Найденные колонки: {fields}")
    
    # --- Создание коллекции ---
    vector_size = model.get_sentence_embedding_dimension()
    
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )
    print(f"Коллекция '{COLLECTION_NAME}' создана (dense={vector_size}d)")
    
    # --- Подготовка точек ---
    points = []
    for idx, (_, row) in enumerate(df.iterrows()):
        payload = {"text": build_search_text(row), "row_index": int(idx)}
        
        # Заполняем поля из CSV
        for field, col_name in fields.items():
            val = row.get(col_name)
            if pd.notna(val):
                if field in ("dn", "pn"):
                    payload[field] = extract_numeric(val)
                else:
                    payload[field] = str(val).strip()
        
        # ---------- ИЗВЛЕКАЕМ БРЕНД ИЗ НАЗВАНИЯ ----------
        if "name" in fields:
            name_val = row.get(fields["name"])
            if pd.notna(name_val):
                brand = extract_brand(str(name_val))
                if brand:
                    payload["brand"] = brand  # перезаписываем, если уже был из CSV (но обычно нет)
        
        search_text = payload["text"]
        dense_vector = model.encode(search_text).tolist()
        
        point = models.PointStruct(
            id=idx,
            vector={
                "dense": dense_vector,
                "sparse": models.Document(text=search_text, model="qdrant/bm25"),
            },
            payload=payload,
        )
        points.append(point)
        
        if len(points) >= 100:
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
            print(f"  Загружено {idx + 1} записей...")
            points = []
    
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    
    print(f"✅ Индексация завершена: {len(df)} записей")


if __name__ == "__main__":
    index_csv("/workspaces/ML-Cancer-Data/DB/mapping_results.csv")  # индексируем все данные