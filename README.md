# Music-Recommendation-API-
This project implements a collaborative filtering music recommendation system using **Python**, **FastAPI**, and **Surprise SVD**. It provides personalized track recommendations based on user listening history.

---

## 📌 **Features**
- Collaborative filtering using SVD matrix factorization (scikit-surprise)
- Implicit ratings derived from play frequency (listening count)
- Normalized ratings on a 1-5 scale using quantiles
- REST API built with FastAPI to serve recommendations in real-time
  
- Clean model loading using FastAPI lifespan

---

## ⚙️ **Tech Stack**
- Python 3.x  
- FastAPI  
- scikit-surprise (SVD)  
- Pandas / NumPy  
- Uvicorn  

---

## 📝 **Data Preprocessing**
The dataset consists of user listening logs:
