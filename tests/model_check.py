# cek_model.py (buat di root folder)
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Model yang tersedia untuk embedding:\n")
for model in genai.list_models():
    if "embed" in model.name.lower():
        print(f"Nama  : {model.name}")
        print(f"Method: {model.supported_generation_methods}")
        print("-" * 40)
