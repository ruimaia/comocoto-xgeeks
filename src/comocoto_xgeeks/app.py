from pathlib import Path
from typing import Any
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fasthx import hx, page, Jinja
from dotenv import load_dotenv

import boto3
import os
import json
import uvicorn


ROOT_DIR = Path(__file__).parent.parent.parent
TEMPLATES = Jinja2Templates(directory=ROOT_DIR / "app")
jinja = Jinja(TEMPLATES)

load_dotenv()
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
S3_CLIENT = boto3.resource("s3")
DATA = None
REQUESTS = None

# Create the app.
app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=ROOT_DIR / "static"),
    name="static",
)

def _fetch_data(file_name) -> list[str, Any]:
    content_object = S3_CLIENT.Object(BUCKET_NAME, file_name)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    return json_content

def render_index(result: list[dict[str, str]], *, context: dict[str, Any], request: Request) -> str:
    html = TEMPLATES.TemplateResponse(
        "index.html", {"request": request}
    )
    return html
        
        
@app.get("/")
@page(render_index)
def index() -> None:
    ...
    
@app.get("/fetch-data")
@jinja.hx("historical_data_list.html")
def fetch_data() -> str:
    global DATA
    DATA = _fetch_data('historic_data.json')
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

@app.get("/new-requests")
@jinja.hx("new_requests.html")
def new_requests():
    global REQUESTS
    REQUESTS = _fetch_data('new_data.json')
    return {"requests": REQUESTS}

@app.post("/train-model")
@jinja.hx("training_results.html")
async def train_model(request: Request):
    form_data = await request.form()
    print(form_data.keys())
    training_metrics = {
        "placeholder1": 1,
        "placeholder2": 2,
        "placeholder3": 3
    }
    return {"training_metrics": training_metrics}
    
@app.get("/generate-budget")
def generate_budget(request_id: int):
    request = REQUESTS[request_id-1]
    return "AQUI esta o budget para o email: " + request["email_body"]
    
if __name__ == "__main__":
    uvicorn.run(app)