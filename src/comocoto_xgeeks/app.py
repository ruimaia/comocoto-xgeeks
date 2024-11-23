from pathlib import Path
from typing import Any
import boto3
from fastapi import FastAPI, Request, Response
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
    data = _fetch_data()
    return {"historical_data": data}
