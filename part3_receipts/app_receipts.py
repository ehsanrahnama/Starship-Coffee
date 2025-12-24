
import re
import json
import os
import base64
from openai import OpenAI
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


st.title("Task 3 — Receipt OCR (Vision Model)")

file = st.file_uploader("Upload receipt (PNG/JPG)", type=["png", "jpg", "jpeg"])



def call_vision_model(file_img):
 
    base64_img = base64.b64encode(file_img.read()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_img}"


    client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)


    prompt = """
            You are reading a shopping receipt.

            Extract the purchased items and the FINAL total amount.
            If a total is crossed out, ignore it and return the current total.

            Return JSON ONLY in this exact format:
            {
            "items": [
                {"name": "...", "qty": 1, "unit_price": "0.00", "line_total": "0.00"}
            ],
            "total": "0.00"
            }
    """

    completion = client.chat.completions.create(
    model="google/gemma-3-27b-it:nebius",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ],
    temperature=0
    )


    raw = completion.choices[0].message.content
    json_text = re.search(r"\{.*\}", raw, re.S).group()

    return json_text

if file:

    print("Processing file:", file)

    result = call_vision_model(file)
    result = json.loads(result)

    st.subheader("JSON Output")
    st.json(result)

    st.subheader("Items Table")
    st.table(result["items"])

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "file": file.name,
        "prediction": result
    }

    with open("part3_receipts/predictions.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
