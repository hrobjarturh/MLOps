import json
import os
import subprocess
import sys

import click
import requests
from PIL import Image

from data.Preprocessing import Preprocessing

IMAGE_SIZE = 224

def is_file(path):
    """
    Check if the given path is a directory, a file, or does not exist.

    Args:
    - path (str): The path to check.

    Returns:
    - 'directory' if the path is a directory.
    - 'file' if the path is a file.
    - 'does not exist' if the path does not exist.
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            return False
        elif os.path.isfile(path):
            return True
    else:
        raise FileNotFoundError

def get_token():
    token = subprocess.check_output(["gcloud","auth", "print-identity-token"])
    return token.decode().strip()

def predict(input_image_path):

    img = Image.open(input_image_path).convert("RGB")

    tensor_data = Preprocessing.preprocess_images(img, IMAGE_SIZE)

    fastapi_url = "https://animage-w3kv235oea-ew.a.run.app"
    endpoint = "/process_tensor"

    # Prepare the payload with the tensor data
    payload = {"data": tensor_data.tolist()}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_token()}",
    }

    # Convert the payload to JSON
    payload_json = json.dumps(payload)

    print('\nSending request ...')
    # Send a POST request to the FastAPI endpoint
    response = requests.post(f"{fastapi_url}{endpoint}", data=payload_json, headers=headers)
    print("Response code:", response.status_code)

    # Check the response
    if response.status_code == 200:
        output = response.json().get("results")
        print("\noutput: ", output)
    elif response.status_code == 401:
        print("Response code:", response.status_code, "\n (Update GCP Bearer token)")
    else:
        print("Error:", response.status_code)

@click.command()
@click.option('--file', default=None, help='Path to file to infere on.')
def argpasser(file):
    if file:
        if not is_file(file):
            print('Input is not a file.')
        else:
            predict(file)
    else:
        print('Please include a path to a file.')

if __name__ == "__main__":

    argpasser()
    
