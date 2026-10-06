import requests
import json

url = "http://127.0.0.1:5001/predict"
data = {"message": "Win a free lottery now"}
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, json=data, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    with open('last_error.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("Saved response to last_error.html")

except Exception as e:
    print(f"Error: {e}")
