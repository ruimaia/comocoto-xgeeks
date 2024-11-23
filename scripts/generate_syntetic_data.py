# %%
import os
from typing import Any

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import concurrent.futures
from tqdm.auto import tqdm
from dotenv import load_dotenv
from jsonargparse import CLI
import boto3
import ast
import json


ROOT_DIR = Path(__file__).parent.parent
JINJA_ENV = Environment(loader=FileSystemLoader(ROOT_DIR / "prompts/"))

load_dotenv()
region = os.getenv("AWS_DEFAULT_REGION")
model_id = "mistral.mistral-large-2407-v1:0"
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=region)
inference_config = {
    "maxTokens": 1028,
    "temperature": 1.0,
    "topP": 0.9
}

def extract_json(resp: str):
    return ast.literal_eval(resp.split("```json")[1].split("```")[0])


def generate_datapoint(prompt: str, required_fields: list[str]) -> dict[str, Any]:
    messages = [
        {
            "role": "user", 
            "content": [
                {"text": prompt}
            ]
        }
    ]
    params = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": inference_config
    }
    resp = bedrock_client.converse(**params)
    try:
        answer = extract_json(resp["output"]["message"]["content"][0]["text"])
        assert all([k in answer for k in required_fields])
    except:
        answer = ""
    return answer

def generate_syntetic_data(
    prompt_name:str, ndatapoints: int, prompt_parameters: dict, output_file: str, required_fields: list[str]    
) -> list[dict[str, Any]]:
    prompt = JINJA_ENV.get_template(prompt_name)
    output_file = ROOT_DIR / output_file
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                generate_datapoint, 
                prompt.render(**prompt_parameters),
                required_fields
            ) 
            for _ in range(ndatapoints)
        ]
        dataset = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=ndatapoints)]
        
    dataset = [datapoint for datapoint in dataset if datapoint]
    with open(output_file, "w") as f:
        json.dump(dataset, f)

if __name__ == "__main__":
    CLI(generate_syntetic_data)