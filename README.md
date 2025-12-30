
```md
````

# Iris Flower Prediction Web Application (PyTorch + Flask + Cloud Run)

This project is a complete end-to-end Machine Learning web application built using **PyTorch** and **Flask**, containerized with **Docker**, and deployed on **Google Cloud Run** using **source-based deployment**.

The application allows users to enter Iris flower measurements through a simple HTML form and returns the predicted Iris species along with confidence probabilities.

---

## 🚀 Features

- PyTorch-based Iris classification model
- Flask backend with HTML & CSS user interface
- Dockerized application
- Serverless deployment using Google Cloud Run
- Scales automatically and charges only on usage
- Beginner-friendly and production-aligned structure

---

## 🧱 Project Structure

````
iris-project-flask/
├── app.py                  # Flask application
├── model.py                # PyTorch model definition
├── predict.py              # Inference logic
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── artifacts/              # Trained model artifacts
│   ├── model.pt
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── templates/
│   └── index.html          # HTML UI
└── static/
└── style.css           # CSS styling

````

---

## 🧠 Model Overview

- **Model Type:** Multi-Layer Perceptron (MLP)
- **Framework:** PyTorch
- **Dataset:** Iris dataset
- **Input Features:**
  - SepalLengthCm
  - SepalWidthCm
  - PetalLengthCm
  - PetalWidthCm
- **Output:** Iris species prediction with probabilities

---

## 🖥️ Run Locally (Without Docker)

### 1. Create virtual environment
```bash
python -m venv venv
````

Activate it:

* **Windows**

```bash
venv\Scripts\activate
```

* **Linux / macOS**

```bash
source venv/bin/activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:8080
```

---

## 🐳 Run Locally Using Docker

### 1. Build Docker image

```bash
docker build -t iris-flask-ui .
```

---

### 2. Run Docker container

```bash
docker run -p 8080:8080 iris-flask-ui
```

Open in browser:

```
http://localhost:8080
```

---

## ☁️ Deploy to Google Cloud Run (Option 1 – Deploy from Source)

This project uses **source-based deployment**, where Google Cloud Build automatically builds the Docker image.

---

### Prerequisites

* Google Cloud account
* Billing enabled
* Google Cloud SDK installed

---

### 1. Authenticate with Google Cloud

```bash
gcloud auth login
```

---

### 2. Set Google Cloud project

```bash
gcloud config set project YOUR_PROJECT_ID
```

To list projects:

```bash
gcloud projects list
```

---

### 3. Enable required services (one-time setup)

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

---

### 4. Deploy to Cloud Run

Run this command from the project root directory:

```bash
gcloud run deploy iris-flask-ui \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated
```

During deployment:

* Accept the default service name
* Select region `europe-west1`
* Allow unauthenticated access → **Yes**

---

### 5. Access the deployed application

After deployment completes, Cloud Run will provide a public URL:

```
https://iris-flask-ui-xxxxx-ew.a.run.app
```

Open the URL in a browser and start using the application.

---

## 📊 Logs & Monitoring

To view logs for the deployed service:

```bash
gcloud run services logs read iris-flask-ui --region europe-west1
```

---

## 💰 Cost Management & Cleanup

Cloud Run charges **only when requests are served**.

### Delete the Cloud Run service

```bash
gcloud run services delete iris-flask-ui --region europe-west1
```

---

### Delete container images (optional but recommended)

```bash
gcloud artifacts repositories delete cloud-run-source-deploy \
  --location=europe-west1
```

After deletion, no compute or storage charges will remain.

---

## 🔐 Production Notes

* Uses `gunicorn` as a production-grade WSGI server
* Binds to `$PORT` environment variable (Cloud Run requirement)
* Stateless architecture suitable for serverless deployment
* Easily extendable for APIs, authentication, or CI/CD

---

## 🔮 Future Improvements

* Add JSON API endpoint (`/api/predict`)
* Input validation
* Authentication
* Model retraining pipeline
* CI/CD integration
* React frontend
* Model versioning

---

## 📄 License

This project is created for **learning and educational purposes**.

---

## 👤 Author

Built as a hands-on learning project to understand:

* Machine Learning deployment
* Flask web applications
* Docker containerization
* Google Cloud Run serverless architecture

```



