from fastapi import FastAPI
import uvicorn
from models import UserSurvey
import time
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

from utils import create_metadata_embedding
from main import recommend
from models import RecommendationResponse, UserSurvey, UserSurveyRequest


load_dotenv()
model = SentenceTransformer(os.getenv('MODEL_NAME'))
painting_embeddings = np.load(os.getenv('EMBEDDING_PATH'))
metadata_path = os.getenv('METADATA_PATH')
if not os.path.exists(painting_embeddings):
    create_metadata_embedding(model, metadata_path, painting_embeddings)


app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "API is running."}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(user_survey: UserSurveyRequest):
    start = time.time()
    profile = UserSurvey(
        age=user_survey.age,
        gender=user_survey.gender,
        artwork_type=user_survey.artwork_type,
        difficulty=user_survey.difficulty
    )
    recommendations = recommend(profile=profile, painting_embeddings=painting_embeddings, model=model,
                                is_new_user=user_survey.is_new_user, top_k=user_survey.top_k, metadata_path=metadata_path)
    
    print(f'Running time: {time.time() - start}')
    return RecommendationResponse(recommendations=recommendations)

if __name__=='__main__':
    # Run the app
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)