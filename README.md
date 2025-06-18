**Resume Skill Clustering and Recommendation Engine** :

---

# **Resume Skill Clustering & Recommendation Engine**
### **A Machine Learning Pipeline for Resume Analysis**

---

## **1. Project Overview**
This project processes **raw resume data**, extracts **skills**, **clusters** resumes into meaningful groups, and provides **skill recommendations** based on gaps in a candidate's profile.

### **Key Objectives:**
1. **Clean & normalize** messy resume data (skills in different formats).
2. **Cluster** resumes into groups (e.g., "Data Scientists," "Web Developers").
3. **Recommend skills** to bridge gaps in a candidate’s profile.
4. **Visualize** clusters for insights.

---

## **2. How It Works**
### **Step 1: Data Preprocessing**
**Input:** Raw resumes (CSV/Excel) with columns like `skills`, `experience`, `job_title`.  
**Output:** Clean, structured data ready for ML.

#### **Key Steps:**
- **Handling missing data** (e.g., filling empty skills with "Not Specified").
- **Skill normalization** (e.g., `"Python3" → "python"`, `"ML" → "machine learning"`).
- **One-hot encoding** (converting skills into binary features).

**Example:**
```python
skills = ["Python, ML, SQL"] → 
cleaned_skills = ["python", "machine learning", "sql"] → 
one_hot_encoded = [1, 1, 1, 0, ...]  # (binary vector)
```

---

### **Step 2: Clustering (K-Means)**
**Goal:** Group resumes into clusters based on skill similarity.

#### **Key Steps:**
1. **Feature Extraction:**  
   - Use **TF-IDF** or **CountVectorizer** on skills.
2. **Dimensionality Reduction (PCA):**  
   - Reduce features for better clustering.
3. **Optimal Cluster Detection:**  
   - Use **Silhouette Score** to find the best number of clusters.
4. **Train K-Means Model:**  
   - Fit model on processed data.

**Output:**  
- Each resume assigned a `cluster_id` (e.g., `0 = Data Scientists`, `1 = Web Devs`).

---

### **Step 3: Skill Recommendations**
**Goal:** Suggest skills to improve a candidate’s profile.

#### **How It Works:**
1. For a given resume, identify its **cluster**.
2. Compare skills against **top skills in that cluster**.
3. Recommend missing skills that are **common in the cluster**.

**Example:**
- **Input Resume Skills:** `["python", "pandas"]`  
- **Cluster 0 Skills:** `["python", "pandas", "tensorflow", "sql"]`  
- **Recommended Skills:** `["tensorflow", "sql"]`  

---

### **Step 4: Visualization**
- **PCA + Scatter Plot:** Show clusters in 2D/3D.
- **Bar Charts:** Top skills per cluster.

**Example:**
```
Cluster 0 (Data Scientists):
- python: 95%
- tensorflow: 80%
- sql: 70%
```

---

## **3. Project Structure**
```
resume-clustering/
├── data/
│   ├── raw/                  # Original data (CSV)
│   └── processed/            # Cleaned data
├── models/                   # Saved ML models
│   ├── kmeans_model.pkl      # Trained K-Means
│   └── skill_encoder.pkl     # Skill vectorizer
├── notebooks/                # Jupyter notebooks
│   ├── 1_Data_Cleaning.ipynb
│   ├── 2_Clustering.ipynb
│   └── 3_Visualization.ipynb
├── scripts/
│   ├── preprocess.py         # Data cleaning
│   ├── train.py              # Model training
│   └── predict.py            # Recommend skills
└── requirements.txt          # Python dependencies
```

---

## **4. How to Use**
### **Installation**
```bash
git clone https://github.com/yourusername/resume-clustering.git
cd resume-clustering
pip install -r requirements.txt
```

### **Run the Pipeline**
1. **Preprocess Data:**
   ```bash
   python scripts/preprocess.py --input data/raw/resumes.csv --output data/processed/clean_resumes.csv
   ```
2. **Train Model:**
   ```bash
   python scripts/train.py --data data/processed/clean_resumes.csv --model models/kmeans_model.pkl
   ```
3. **Get Recommendations:**
   ```python
   from scripts.predict import recommend_skills
  
   skills = ["python", "sql"]
   recommendations = recommend_skills(skills, model_path="models/kmeans_model.pkl")
   print(recommendations)
   ```

**Output Example:**
```json
{
  "cluster": 0,
  "cluster_name": "Data Scientists",
  "recommended_skills": ["tensorflow", "pytorch"],
  "similar_profiles": [
    {"job_title": "Data Engineer", "skills": "python, sql, spark"},
    {"job_title": "ML Researcher", "skills": "python, tensorflow, numpy"}
  ]
}
```

---

## **5. Key Customizations**
- **Add More Skills:** Update `skill_synonyms` in `preprocess.py`.
- **Change Clustering Algorithm:** Modify `train.py` (try DBSCAN, HDBSCAN).
- **Deploy as API:** Use FastAPI to wrap `predict.py`.

---

## **6. Why This Project?**
✅ **HR Tech:** Auto-categorize resumes for recruiters.  
✅ **Career Growth:** Skill recommendations for professionals.  
✅ **Job Matching:** Find similar candidates.  
✅ **You can Analyze** The job discription that you fit on this perticular job or not.
---

## **7. Future Improvements**
- **Add NLP:** Extract skills from raw text resumes.
- **Skill Graphs:** Visualize skill relationships.
- **Deep Learning:** Use embeddings for better clustering.
