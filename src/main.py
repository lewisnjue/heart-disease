from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from config import get_settings

SETTINGS = get_settings()

app = FastAPI()

app.mount("/static", StaticFiles(directory=SETTINGS.BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=SETTINGS.BASE_DIR / "templates")

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")  

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="main.html", context={}
    )

@app.post("/predict", response_class=JSONResponse)
async def predict(
    age: float = Form(...),
    sex: float = Form(...),
    cp: float = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: float = Form(...),
    restecg: float = Form(...),
    thalach: float = Form(...),
    exang: float = Form(...),
    oldpeak: float = Form(...),
    slope: float = Form(...),
    ca: float = Form(...),
    thal: float = Form(...),
):
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    df = pd.DataFrame([data])

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)

    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"

    return JSONResponse(content={"result": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)