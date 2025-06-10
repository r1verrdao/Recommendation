from pydantic import BaseModel
from typing import List

# Schema cho dữ liệu đầu vào để embedding profile người dùng
class UserSurvey(BaseModel):
    age: str
    gender: str
    artwork_type: str
    difficulty: str


# Schema cho dữ liệu đầu vào của API
class UserSurveyRequest(BaseModel):
    age: str
    gender: str
    artwork_type: str
    difficulty: str
    is_new_user: bool = False
    top_k: int = 5

# Schema cho dữ liệu trả về của API
class RecommendationResponse(BaseModel):
    recommendations: List[str]