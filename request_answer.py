import requests
from constants import API_KEY

def generate_content(text):
    api_key = API_KEY

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": text
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

