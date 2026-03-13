"""Fetch Dartmouth Chat API models and write to models.txt."""
import os

import requests
from dotenv import load_dotenv


load_dotenv()

key = os.environ.get("DARTMOUTH_CHAT_API_KEY")
if not key:
    print('Set up DARTMOUTH_CHAT_API_KEY in .env!')
    exit(-1)

resp = requests.get(
    "https://chat.dartmouth.edu/api/models",
    headers={"Authorization": f"Bearer {key}"},
)
resp.raise_for_status()
data = resp.json()
models = [m["id"] for m in data.get("data", [])]

with open('models.txt', 'w+') as f:
    f.writelines('\n'.join(models))

print(f'Wrote {len(models)} models.')