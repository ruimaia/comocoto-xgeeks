from pathlib import Path
from typing import Any
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fasthx import hx, page, Jinja

ROOT_DIR = Path(__file__).parent.parent.parent
TEMPLATES = Jinja2Templates(directory=ROOT_DIR / "app")

# Create the app.
app = FastAPI()

def render_index(result: list[dict[str, str]], *, context: dict[str, Any], request: Request) -> str:
    html = TEMPLATES.get_template("index.html").render()
    return html
        
        
@app.get("/")
@page(render_index)
def index() -> None:
    ...
