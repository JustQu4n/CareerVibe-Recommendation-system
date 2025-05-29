from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId
import os
from dotenv import load_dotenv
from parse_cv import parse_cv 
# from pydantic import BaseModel
# from transformers import pipeline

load_dotenv()

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối đến MongoDB Atlas
MONGO_URI = os.getenv('DATABASE_URL')
client = MongoClient(MONGO_URI)
db = client["careervibe_db"]
jobs_collection = db["jobposts"]
jobseekers_collection = db["jobseekers"]

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# class JDInput(BaseModel):
#     text: str

# @app.post("/api/summarize-jd")
# def summarize_jd(data: JDInput):
#     summary = summarizer(data.text, max_length=100, min_length=30, do_sample=False)
#     return {"summary": summary[0]['summary_text']}

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

    
@app.post("/match-cv")
async def match_cv_with_jobs(file: UploadFile = File(...)):
    """
    Upload a CV and get matching job recommendations
    """
    # Kiểm tra file type
    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                     "application/docx"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Only PDF and DOCX files are supported."
        )
    
    # Đọc file
    file_bytes = await file.read()
    
    try:
        # Parse CV để trích xuất thông tin
        cv_data = parse_cv(file_bytes, file.content_type)
        
        # Lấy danh sách công việc từ MongoDB
        jobs = list(jobs_collection.find({"status": "active"}))
        
        # Tính toán matching score
        job_scores = []
        
        for job in jobs:
            # Extract job features
            job_skills = job.get("skills", [])
            
            # Ensure job_experience is an integer
            try:
                job_experience = int(job.get("experience", 0))
            except (ValueError, TypeError):
                job_experience = 0
                
            job_industries = job.get("industries", [])
            
            # Calculate skill match score (50% weight)
            skill_matches = sum(1 for skill in cv_data["skills"] if skill.lower() in [s.lower() for s in job_skills])
            max_skills = max(len(cv_data["skills"]), len(job_skills))
            skill_score = (skill_matches / max_skills) if max_skills > 0 else 0
            
            # Calculate experience match score (30% weight)
            # Ensure cv_data experience is an integer
            cv_experience = int(cv_data["experience"]) if isinstance(cv_data["experience"], (int, str)) else 0
            exp_diff = abs(cv_experience - job_experience)
            exp_score = 1.0 / (1.0 + exp_diff)  # Closer to 1 means better match
            
            # Calculate industry match score (20% weight)
            industry_matches = sum(1 for ind in cv_data["industries"] if ind.lower() in [i.lower() for i in job_industries])
            max_industries = max(len(cv_data["industries"]), len(job_industries))
            industry_score = (industry_matches / max_industries) if max_industries > 0 else 0
            
            # Calculate weighted score
            total_score = (skill_score * 0.5) + (exp_score * 0.3) + (industry_score * 0.2)
            
            job_scores.append({
                "job_id": str(job["_id"]),
                "title": job["title"],
                "company": job.get("company", ""),
                "score": round(total_score, 2),
                "skill_match": round(skill_score, 2),
                "experience_match": round(exp_score, 2),
                "industry_match": round(industry_score, 2)
            })
        
        # Sort by score
        job_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 10 matches
        return {
            "cv_info": {
                "skills": cv_data["skills"],
                "experience": cv_data["experience"],
                "industries": cv_data["industries"]
            },
            "matches": job_scores[:10]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")
    

@app.post("/match-cv-to-job/{job_id}")
async def match_cv_to_specific_job(job_id: str, file: UploadFile = File(...)):
    """
    Upload a CV and get matching score with a specific job post
    """
    # Kiểm tra file type
    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                     "application/docx"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Only PDF and DOCX files are supported."
        )
    
    try:
        # Kiểm tra xem job_id có tồn tại không
        job = jobs_collection.find_one({"_id": ObjectId(job_id), "status": "active"})
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job post with ID {job_id} not found or inactive"
            )
        
        # Đọc file
        file_bytes = await file.read()
        
        # Parse CV để trích xuất thông tin
        cv_data = parse_cv(file_bytes, file.content_type)
        
        # Extract job features
        job_skills = job.get("skills", [])
        
        # Ensure job_experience is an integer
        try:
            job_experience = int(job.get("experience", 0))
        except (ValueError, TypeError):
            job_experience = 0
            
        job_industries = job.get("industries", [])
        
        # Calculate skill match score (50% weight)
        skill_matches = sum(1 for skill in cv_data["skills"] if skill.lower() in [s.lower() for s in job_skills])
        max_skills = max(len(cv_data["skills"]), len(job_skills))
        skill_score = (skill_matches / max_skills) if max_skills > 0 else 0
        
        # Calculate experience match score (30% weight)
        # Ensure cv_data experience is an integer
        cv_experience = int(cv_data["experience"]) if isinstance(cv_data["experience"], (int, str)) else 0
        exp_diff = abs(cv_experience - job_experience)
        exp_score = 1.0 / (1.0 + exp_diff)  # Closer to 1 means better match
        
        # Calculate industry match score (20% weight)
        industry_matches = sum(1 for ind in cv_data["industries"] if ind.lower() in [i.lower() for i in job_industries])
        max_industries = max(len(cv_data["industries"]), len(job_industries))
        industry_score = (industry_matches / max_industries) if max_industries > 0 else 0
        
        # Calculate weighted score
        total_score = (skill_score * 0.5) + (exp_score * 0.3) + (industry_score * 0.2)
        
        # Create detailed matching analysis
        matched_skills = [skill for skill in cv_data["skills"] if skill.lower() in [s.lower() for s in job_skills]]
        missing_skills = [skill for skill in job_skills if skill.lower() not in [s.lower() for s in cv_data["skills"]]]
        
        matched_industries = [ind for ind in cv_data["industries"] if ind.lower() in [i.lower() for i in job_industries]]
        # Fix here: changed i.lower() to s.lower()
        missing_industries = [ind for ind in job_industries if ind.lower() not in [s.lower() for s in cv_data["industries"]]]
        
        # Return detailed matching result
        return {
            "job_details": {
                "job_id": str(job["_id"]),
                "title": job["title"],
                "company": job.get("company", ""),
                "required_skills": job_skills,
                "required_experience": job_experience,
                "industries": job_industries
            },
            "cv_info": {
                "skills": cv_data["skills"],
                "experience": cv_experience,
                "industries": cv_data["industries"]
            },
            "matching_scores": {
                "overall_score": round(total_score * 100, 1),  # Convert to percentage
                "skill_match": round(skill_score * 100, 1),
                "experience_match": round(exp_score * 100, 1),
                "industry_match": round(industry_score * 100, 1)
            },
            "matching_details": {
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "matched_industries": matched_industries,
                "missing_industries": missing_industries,
                "experience_difference": exp_diff
            }
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")