import ast
import json

def extract_json(resp: str):
    return json.loads(resp.split("```json")[1].split("```")[0])