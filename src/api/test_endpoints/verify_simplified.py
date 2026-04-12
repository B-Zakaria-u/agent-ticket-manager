import os
import sys
from dotenv import load_dotenv

sys.path.append(os.getcwd())
load_dotenv()

from fastapi.testclient import TestClient
from src.api.app import create_app

app = create_app()
client = TestClient(app)

def test_simplified_endpoints():
    endpoints = ["/test/spec", "/test/coding", "/test/testing", "/test/scout"]
    
    for ep in endpoints:
        print(f"\n--- Testing {ep} (Empty Payload) ---")
        # Sending an empty POST request (no body)
        response = client.post(ep, json={})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Agent Status: {data['status']}")
            print(f"Summary: {data['summary']}")
            # Check if defaults were applied
            if ep == "/test/spec":
                spec_len = len(data['state_updates'].get('spec', ''))
                print(f"Spec generated? {'Yes' if spec_len > 0 or data['status'] == 'failed' else 'No'}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    test_simplified_endpoints()
