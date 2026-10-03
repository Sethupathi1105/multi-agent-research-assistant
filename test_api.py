import requests
import time

print("Sending request to /research...")
start = time.time()

response = requests.post(
    "http://127.0.0.1:8000/research",
    json={"question": "What is CrewAI used for?"}
)

elapsed = time.time() - start
print(f"\nResponse received after {elapsed:.1f} seconds")
print(f"Status code: {response.status_code}")
print("\n--- RESPONSE BODY ---")
print(response.json())