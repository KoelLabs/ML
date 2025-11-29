import os
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

model = "google"  # or "deepspeech"

# url = f"http://127.0.0.1:5000/api/v1/asr/{model}"
url = f"https://koel-api.fly.dev/api/v1/asr/{model}"
headers = {"Authorization": f"Bearer {os.environ['API_KEY']}"}

files = {"audio_file": open("data/ExamplesWithComments/TIMIT_sample_0.wav", "rb")}

response = requests.post(url, headers=headers, files=files)

print(response.status_code)
print(response.text)
