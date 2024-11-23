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

from comocoto_xgeeks.bedrock import generate_datapoint


ROOT_DIR = Path(__file__).parent.parent
JINJA_ENV = Environment(loader=FileSystemLoader(ROOT_DIR / "prompts/"))

load_dotenv()
region = os.getenv("AWS_DEFAULT_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


s3_client = boto3.resource("s3")

model_id = "mistral.mistral-large-2407-v1:0"
inference_config = {
    "maxTokens": 1028,
    "temperature": 1.0,
    "topP": 0.9
}

def generate_syntetic_data(
    prompt_name:str, ndatapoints: int, prompt_parameters: dict, output_file: str, required_fields: list[str]    
) -> list[dict[str, Any]]:
    prompt = JINJA_ENV.get_template(prompt_name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                generate_datapoint, 
                model_id,
                prompt.render(**prompt_parameters),
                required_fields,
                inference_config
            ) 
            for _ in range(ndatapoints)
        ]
        dataset = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=ndatapoints)]
        
    dataset = [datapoint for datapoint in dataset if datapoint]
    
    s3object = s3_client.Object(BUCKET_NAME, output_file)
    s3object.put(
        Body=(bytes(json.dumps(dataset).encode('UTF-8')))
    )
    # with open(output_file, "w") as f:
    #     json.dump(dataset, f)

if __name__ == "__main__":
    CLI(generate_syntetic_data)