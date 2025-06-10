import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
import time
from dotenv import load_dotenv
import os

from utils import get_user_profile_embedding, create_metadata_embedding
from models import UserSurvey


def recommend(
    profile: UserSurvey,
    painting_embeddings: np.ndarray,
    model: SentenceTransformer,
    metadata_path: str,
    is_new_user: bool = False,
    top_k: int = 5
) -> List[str]:
    """
    Đưa ra danh sách các tranh được đề xuất dựa trên hồ sơ người dùng.

    Args:
        profile (UserSurvey): Thông tin khảo sát người dùng.
        painting_embeddings (np.ndarray): Embedding của các tranh.
        model (SentenceTransformer): Mô hình SBERT cho embedding.
        metadata_path (str): Đường dẫn file metadata.
        is_new_user (bool): Cờ xác định người dùng mới.
        top_k (int): Số lượng tranh đề xuất.

    Returns:
        List[str]: Danh sách ID các tranh được đề xuất.
    """
    profile_embedding: np.ndarray = get_user_profile_embedding(profile, model)
    similarities: np.ndarray = cosine_similarity([profile_embedding], painting_embeddings)[0]
    top_k_indices: np.ndarray = similarities.argsort()[-top_k:][::-1]

    with open(metadata_path, 'r', encoding='utf-8') as f:
        paintings = json.load(f)

    recommendations: List[str] = [paintings[i]['id'] for i in top_k_indices]

    return recommendations



# if __name__=='__main__':

#     load_dotenv()

#     model = SentenceTransformer(os.getenv('MODEL_NAME'))
#     painting_embeddings = np.load(os.getenv('EMBEDDING_PATH'))
#     metadata_path = os.getenv('METADATA_PATH')
#     if not os.path.exists(painting_embeddings):
#         create_metadata_embedding(model, metadata_path, painting_embeddings)

#     results




    # test_survey_data = {
    # "age": "25-",
    # "gender": "Female",
    # "artwork_type": "Creatures, Fusion, Magic",
    # "difficulty": "Easy"
    # }
    
    # test_survey = UserSurvey(**test_survey_data)

    # # create_metadata_embedding()
    # results = recommend(profile=test_survey, painting_embeddings=painting_embeddings, model=model)
    # print(results)

    # print(f'Running time: {time.time()- start}')