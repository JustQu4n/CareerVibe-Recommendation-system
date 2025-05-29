import PyPDF2
import docx
import re
from io import BytesIO

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file bytes"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file bytes"""
    doc = docx.Document(BytesIO(file_bytes))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_cv(file_bytes, file_type):
    """Extract text based on file type"""
    if file_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/docx"]:
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def extract_skills(text):
    """Extract skills from CV text"""
    # Common programming languages and technologies
    common_skills = [
        "python", "java", "javascript", "js", "typescript", "ts", "c\\+\\+", "c#", "ruby", "php", 
        "html", "css", "react", "angular", "vue", "node", "express", "django", "flask", "spring", "reactjs", "nextjs",
        "graphql", "sql", "nosql", "rest", "api", "web development", "full stack", "backend", "frontend",
        "nodejs", "expressjs", "django", "flask", "spring boot", "laravel", "vuejs", "angularjs",
        "html5", "css3", "sass", "less", "bootstrap", "tailwind", "material ui", "responsive web design",
        "web design", "ui", "ux", "user interface", "user experience", "responsive design", "mobile first",
        "bootstrap", "jquery", "sql", "mysql", "postgresql", "mongodb", "firebase", "aws", "azure", 
        "flutter", "dart", "swift", "kotlin", "objective c", "react native", "ionic", "cordova",
        "gcp", "docker", "kubernetes", "git", "github", "rest api", "graphql", "json", "xml",
        "machine learning", "data science", "ai", "artificial intelligence", "nlp", "data analysis",
        "excel", "powerpoint", "word", "communication", "teamwork", "leadership", "problem solving",
        "project management", "agile", "scrum", "jira", "confluence", "devops", "ci/cd", "testing",
        "selenium", "cypress", "unit testing", "integration testing", "performance testing", "security",
        "cybersecurity", "networking", "cloud computing", "big data", "etl", "data warehousing",
        "business intelligence", "tableau", "power bi", "data visualization", "seo", "sem", "digital marketing",
        "content marketing", "social media", "email marketing", "salesforce", "crm", "erp", "sap",
        "human resources", "hr", "recruitment", "talent acquisition", "payroll", "employee relations",
        "customer service", "support", "help desk", "it support", "technical support", "network administration",
        "system administration", "database administration", "sql server", "oracle", "mongodb", "redis",
        "linux", "unix", "windows", "macos", "ios", "android", "mobile development", "web development",
        "full stack development", "backend development", "frontend development", "ui/ux design",
        "graphic design", "photoshop", "illustrator", "figma", "adobe xd", "video editing", "premiere pro",
        "after effects", "animation", "3d modeling", "blender", "autocad", "solidworks", "game development",
        "unity", "unreal engine", "agile methodologies", "scrum master", "kanban", "lean", "waterfall",
        "business analysis", "data engineering", "etl processes", "data pipelines", "data governance",
        "data privacy", "compliance", "gdpr", "hipaa", "pci dss", "risk management", "incident response",
        "penetration testing", "vulnerability assessment", "threat modeling", "security architecture",
        "cloud security", "application security", "network security", "endpoint security", "identity and access management",
        "iam", "zero trust", "devsecops", "security operations center", "soc", "incident management",
        "business continuity", "disaster recovery", "it governance", "it service management", "itsm",
        "it asset management", "configuration management", "change management", "release management",
    ]
    
    # Create regex pattern to find skills
    pattern = r'\b(?:' + '|'.join(common_skills) + r')\b'
    found_skills = re.findall(pattern, text.lower())
    
    # Remove duplicates and return
    return list(set(found_skills))

def extract_experience(text):
    """Extract years of experience from CV text"""
    # Look for patterns like "X years of experience" or "X+ years"
    patterns = [
        r'(\d+)\+?\s*(?:years|yrs)(?:\s+of)?\s+experience',
        r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years|yrs)',
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years.extend([int(y) for y in matches])
    
    # Return the maximum years found or 0 if none found
    return max(years) if years else 0

def extract_industries(text):
    """Extract industries from CV text"""
    common_industries = [
        "software development", "web development", "mobile development", "data science",
        "machine learning", "artificial intelligence", "cloud computing", "devops",
        "cybersecurity", "network", "database", "business intelligence", "data analysis",
        "marketing", "sales", "customer service", "finance", "accounting", "human resources",
        "healthcare", "education", "retail", "manufacturing", "logistics", "transportation",
        "hospitality", "tourism", "media", "entertainment", "telecommunications"
    ]
    
    found_industries = []
    for industry in common_industries:
        if industry.lower() in text.lower():
            found_industries.append(industry)
    
    return found_industries

def parse_cv(file_bytes, file_type):
    """Parse CV and extract relevant information"""
    text = extract_text_from_cv(file_bytes, file_type)
    
    skills = extract_skills(text)
    experience = extract_experience(text)
    industries = extract_industries(text)
    
    return {
        "text": text,
        "skills": skills,
        "experience": experience,
        "industries": industries
    }