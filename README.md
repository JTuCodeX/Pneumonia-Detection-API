# 🩺 Pneumonia Detection API

A FastAPI-based inference service for pneumonia detection using chest X-ray images.  
This API loads a **trained ResNet18 model** and provides endpoints for **image upload** and **prediction**, returning JSON results with predicted class (`NORMAL` / `PNEUMONIA`) and confidence scores.  
Dockerized for easy deployment on Render, Heroku, or AWS.

---

## 🚀 Features
- 🔬 **Deep Learning Model** – ResNet18 trained on chest X-ray dataset.
- ⚡ **FastAPI Backend** – High-performance REST API.
- 📂 **Image Upload Support** – Upload `.jpg` or `.png` files.
- 📊 **JSON Output** – Prediction and confidence score.
- 🐳 **Dockerized** – Portable, production-ready.
- 🌍 **Deployed on Render** – Accessible via web UI and API endpoints.

---

## 🏗️ Project Structure

pneumonia-detection-api/
│── app/
│ ├── main.py # FastAPI entry point
│ ├── inference.py # Model loading & prediction
│ ├── templates/ # Frontend (upload UI)
│ └── utils.py # Preprocessing & helpers
│
│── models/
│ └── pneumonia_resnet18.pth # Trained model weights
│
│── requirements.txt
│── Dockerfile
│── docker-compose.yml
│── README.md





## 🔧 Installation (Local Development)

1. **Clone the repo**  
```bash
git clone https://github.com/<your-username>/pneumonia-detection-api.git
cd pneumonia-detection-api
```


2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the API**

```bash
uvicorn app.main:app --reload
```

4. **Test endpoints**
Web UI → http://127.0.0.1:8000


# 🐳 Run with Docker

1. **Build image**

```bash
docker build -t pneumonia-api .
```

2. **Run container**

```bash
docker run -p 8000:8000 pneumonia-api
```

3. **Open in browser:**

Web UI → http://127.0.0.1:8000/



# ☁️ Deployment on Render

- Push repo to GitHub.

- Create a new Web Service on Render

- Select Docker environment.

- Expose port 8000.

- Done ✅

# 📸 Example Prediction
```bash
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9725
}
```