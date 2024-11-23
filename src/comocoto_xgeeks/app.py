from pathlib import Path
from typing import Any
from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fasthx import hx, page, Jinja
from dotenv import load_dotenv

import boto3
import os
import json
from jinja2 import Environment, FileSystemLoader
import uvicorn

from comocoto_xgeeks.bedrock import generate_datapoint


ROOT_DIR = Path(__file__).parent.parent.parent
TEMPLATES = Jinja2Templates(directory=ROOT_DIR / "app")
JINJA = Jinja(TEMPLATES)
PROMPT_ENV = Environment(loader=FileSystemLoader(ROOT_DIR / "prompts/"))
PARSE_REQUEST_PROMPT = PROMPT_ENV.get_template("parse_request.j2")

load_dotenv()
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
S3_CLIENT = boto3.resource("s3")
DATA = None
REQUESTS = None
SELECTED_ATTRIBUTES = None

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


def extract_attributes(type_of_business: str, feature_list: list[str], request_text: str) -> list[dict[str, Any]]:
    prompt = PARSE_REQUEST_PROMPT.render(
        type_of_business=type_of_business, 
        feature_list="\n".join(feature_list), 
        request_text=request_text
    )
    result = generate_datapoint(
        "mistral.mistral-large-2407-v1:0",
        prompt,
        {
            "maxTokens": 1028,
            "temperature": 0.0,
            "topP": 0.9
        },
        ["characteristics"]
    )
    return result
        
        
@app.get("/")
@page(render_index)
def index() -> None:
    ...
    
@app.get("/fetch-data")
@JINJA.hx("historical_data_list.html")
def fetch_data() -> str:
    global DATA
    DATA = _fetch_data('historic_data.json')
    return {"historical_data": DATA}

@app.get("/popup-form")
@JINJA.hx("popup_form.html")
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
@JINJA.hx("training_results.html")
async def train_model(request: Request):
    global SELECTED_ATTRIBUTES
    form_data = await request.form()
    SELECTED_ATTRIBUTES = form_data.keys()
    training_metrics = {
        "placeholder1": 1,
        "placeholder2": 2,
        "placeholder3": 3
    }
    return {"training_metrics": training_metrics}

@app.get("/new-requests")
@JINJA.hx("new_requests.html")
def new_requests():
    global REQUESTS
    REQUESTS = _fetch_data('new_data.json')
    return {"requests": REQUESTS}
    
@app.get("/generate-budget", response_class=PlainTextResponse)
def generate_budget(request_id: int):
    request = REQUESTS[request_id-1]
    parsed_attributes = extract_attributes(
        "window framing", 
        SELECTED_ATTRIBUTES, 
        request["email_body"]
    )
    
    return json.dumps(parsed_attributes).replace("\n", "<br>")
    
if __name__ == "__main__":
    uvicorn.run(app)