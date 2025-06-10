from models import UserSurvey
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import List

def create_painting_text(painting):
    fields = [
        painting.get('art_genre', ''),
        painting.get('description', ''),
        painting.get('background', ''),
        painting.get('feeling', ''),
        painting.get('category', ''),
        painting.get('difficulty', '')
    ]
    return ', '.join(field for field in fields if field)


def get_user_profile_embedding(survey: UserSurvey, model: SentenceTransformer) -> np.ndarray:
    
    user_profile: str = f"{survey.age}, {survey.gender}, {survey.artwork_type}, {survey.difficulty}"
    embedding: np.ndarray = model.encode([user_profile], convert_to_numpy= True)
    return embedding[0]



def create_metadata_embedding(
    model: SentenceTransformer, 
    metadata_path: str, 
    output_path: str) -> None:
    """
    Tạo và lưu embedding cho metadata của tranh vẽ.

    Args:
        model (SentenceTransformer): Mô hình SBERT dùng để encode text.
        metadata_path (str): Đường dẫn file JSON chứa metadata.
        output_path (str): Đường dẫn file .npy để lưu embedding.

    Returns:
        None
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        paintings = json.load(f)
    painting_texts: List[str] = [create_painting_text(p) for p in paintings]
    embeddings: np.ndarray = model.encode(painting_texts, convert_to_numpy=True)
    np.save(output_path, embeddings)