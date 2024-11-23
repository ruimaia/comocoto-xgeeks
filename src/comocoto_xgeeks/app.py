from pathlib import Path
from typing import Any
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fasthx import hx, page, Jinja

import json

ROOT_DIR = Path(__file__).parent.parent.parent
TEMPLATES = Jinja2Templates(directory=ROOT_DIR / "app")
jinja = Jinja(TEMPLATES)

with open(ROOT_DIR / "data" / "historic_data.json") as f:
    DATA = json.load(f)

# Create the app.
app = FastAPI()

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
    return {"historical_data": DATA}
