from typing import Any, Optional
from dotenv import load_dotenv

import boto3
import os

from comocoto_xgeeks.utils import extract_json

load_dotenv()
region = os.getenv("AWS_DEFAULT_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


BEDROCK_CLIENT = boto3.client(service_name='bedrock-runtime', region_name=region)


def generate_datapoint(
    model_id: str, prompt: str, inference_config: dict[str, Any], required_fields: Optional[list[str]] = None
) -> dict[str, Any]:
    required_fields = required_fields or []
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
    resp = BEDROCK_CLIENT.converse(**params)
    try:
        answer = extract_json(resp["output"]["message"]["content"][0]["text"])
        assert all([k in answer for k in required_fields])
    except:
        print(resp["output"]["message"]["content"][0]["text"])
        answer = {}
    return answer


