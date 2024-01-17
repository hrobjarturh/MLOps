import json
import subprocess

import requests
from PIL import Image

from data.Preprocessing import Preprocessing

IMAGE_SIZE = 224


if __name__ == "__main__":
    input_image_path = "dog.png"
    img = Image.open(input_image_path).convert("RGB")

    tensor_data = Preprocessing.preprocess_images(img, IMAGE_SIZE)

    fastapi_url = "https://animage-w3kv235oea-ew.a.run.app"
    endpoint = "/process_tensor"

    # Prepare the payload with the tensor data
    payload = {"data": tensor_data.tolist()}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjFmNDBmMGE4ZWYzZDg4MDk3OGRjODJmMjVjM2VjMzE3YzZhNWI3ODEiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTAwMzg0NjY3MjI5MzExMzA0ODg4IiwiZW1haWwiOiJocm9iamFydHVyaEBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXRfaGFzaCI6InFZVTJJN1JEclVfSXZuaVM1bFJVM2ciLCJpYXQiOjE3MDU0ODQ4ODcsImV4cCI6MTcwNTQ4ODQ4N30.NnglN1m_zUTfutbKsMeqalEG6c1CFikXYqpqbahis9-la2GEY-Bt-qsGByW6gVri5jEZovvXN7aQn2dQLbKZ1y5WiFO75Gc48GFZ2sr5w1eIz_js-itEQ9pvi8lBmDubxgynO8aqAZWFJniDLqSkRQb-rGsLHMaTWiO7SvKBeQk5Rz0ncJObHMYArLsijiR7ByC9pQ92yTFHmMMdJ2DWsPZKwQYmODJuCRNgKOGX6Ib_9lR82yy-vni_mF25Wtmsn26mC7zHBWrxba-WwEjH7e6l0EpVe7lJH7rfFNM2oIpe0-BlkPlNO1K-sfGq6auM_W04Pu65lqfUq3DIutnM9g",
    }

    # Convert the payload to JSON
    payload_json = json.dumps(payload)

    # Send a POST request to the FastAPI endpoint
    response = requests.post(f"{fastapi_url}{endpoint}", data=payload_json, headers=headers)

    # Check the response
    if response.status_code == 200:
        print(response.status_code)
        output = response.json().get("results")
        print("output: ", output)
    elif response.status_code == 401:
        print("Response code:", response.status_code, "\n (Update GCP Bearer token)")
    else:
        print("Error:", response.status_code)
