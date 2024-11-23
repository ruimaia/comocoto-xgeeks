from pathlib import Path
from typing import Any
import boto3
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fasthx import hx, page, Jinja
from dotenv import load_dotenv

import os
import json

ROOT_DIR = Path(__file__).parent.parent.parent
TEMPLATES = Jinja2Templates(directory=ROOT_DIR / "app")
jinja = Jinja(TEMPLATES)

load_dotenv()
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
S3_CLIENT = boto3.resource("s3")
DATA = None

# Create the app.
app = FastAPI()

def _fetch_data() -> list[str, Any]:
    content_object = S3_CLIENT.Object(BUCKET_NAME, 'historic_data.json')
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    return json_content

def render_index(result: list[dict[str, str]], *, context: dict[str, Any], request: Request) -> str:
    html = TEMPLATES.get_template("index.html").render()
    return html
        
        
@app.get("/")
@page(render_index)
def index() -> None:
    ...
    
@app.get("/fetch-data")
@jinja.hx("historical_data_list.html")
def fetch_data() -> str:
    global DATA
    DATA = _fetch_data()
    return {"historical_data": DATA}

@app.get("/popup-form")
@jinja.hx("popup_form.html")
def popup_form():
    attributes = set([
        attribute
        for data in DATA
        for product in data["caracteristics"]
        for attribute in product.keys()
        if attribute not in ["budget"]
    ])
    return {"attributes": attributes}

@app.post("/train-model")
def train_model(request: Request):
    result = request.headers.get('Hx-Prompt')
    print(result)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)