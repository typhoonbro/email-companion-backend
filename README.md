# Email Companion Backend 🧪

Backend for the Email Companion (https://github.com/typhoonbro/email-companion-frontend), built with **Python**.  
A simple **FastAPI REST API** serving endpoints for the frontend prototype.

---

## **Technologies**

- Python 3.11+  
- FastAPI  
- Uvicorn  
- python-dotenv  

---

## **Main Features**

- **RESTful Endpoints** – Handles GET, POST, and other routes  
- **CORS Enabled** – Works with frontend on different domains  
- **No Database Required** – Stateless API  
- **Easy Deployment** – Ready for Render, Railway, or Heroku  

---

## **How to Run Locally**

```bash
git clone https://github.com/typhoonbro/teste-autoU-backend.git
cd teste-autoU-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## **Access it on the browser**
- http://localhost:8000
You should see a welcome message.

---

## **Next Steps / Improvements**

- Add more endpoints for frontend features

- Unit and integration testing

- Performance improvements and API documentation



