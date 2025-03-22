from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os 
from config import get_settings 
SETTINGS = get_settings()

app = FastAPI()

templates = Jinja2Templates(directory=SETTINGS.BASE_DIR/'templates')


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="main.html", context={}
    )