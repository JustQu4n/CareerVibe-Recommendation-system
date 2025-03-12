from fastapi import FastAPI
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Kết nối đến MongoDB Atlas
MONGO_URI = os.getenv('DATABASE_URL')
client = MongoClient(MONGO_URI)
db = client["careervibe_db"]
jobs_collection = db["jobposts"]
jobseekers_collection = db["jobseekers"]

@app.get("/recommend/{user_id}")
def recommend_jobs(user_id: str):
    # Lấy thông tin ứng viên
    user = jobseekers_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return {"message": "User not found"}

    user_skills = user.get("skills", [])

    # Lấy danh sách công việc
    jobs = list(jobs_collection.find({"status": "active"}))

    job_texts = []
    job_ids = []

    for job in jobs:
        skills = job.get("skills", [])  # Lấy danh sách kỹ năng từ Job
        job_text = f"{job['title']} {' '.join(skills)}"
        job_texts.append(job_text)
        job_ids.append(job["_id"])

    # Xử lý TF-IDF
    user_profile = " ".join(user_skills)  # Hồ sơ ứng viên
    texts = [user_profile] + job_texts  # Ghép ứng viên với danh sách công việc

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Sắp xếp theo độ tương đồng
    recommended_jobs = sorted(zip(job_ids, similarities), key=lambda x: x[1], reverse=True)

    # Trả về danh sách công việc phù hợp nhất
    results = []
    for job_id, score in recommended_jobs[:5]:  # Lấy 5 công việc tốt nhất
        job = jobs_collection.find_one({"_id": job_id})
        results.append({
            "job_id": str(job["_id"]),
            "title": job["title"],
            "score": round(float(score), 2)
        })

    return {"recommendations": results}

@app.get("/debug/users")
def debug_users():
    # Get total count
    user_count = jobseekers_collection.count_documents({})
    
    # Get a sample user to examine the structure
    sample_user = jobseekers_collection.find_one()
    if sample_user:
        # Convert ObjectId to string for JSON serialization
        if '_id' in sample_user:
            sample_user['_id'] = str(sample_user['_id'])
        
        # Extract keys to understand the document structure
        user_keys = list(sample_user.keys())
        
        return {
            "total_users": user_count,
            "sample_user_keys": user_keys,
            "id_field_example": user_keys[0] if user_keys else None
        }
    
    return {"message": "No users found in database"}