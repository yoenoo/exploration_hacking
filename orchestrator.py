import requests

# set your API key
api_key = "YOUR_API_KEY_HERE"

# worker endpoint
endpoint_id = "krg4msxgm9b15u"

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

cuda_code = """
#include <stdio.h>
__global__ void kernel(float *x) { int i = threadIdx.x; x[i] = i*i; }
int main() { float x[256]; kernel<<<1,256>>>(x); cudaDeviceSynchronize(); return 0; }
"""

data = {
  "input": {"code": cuda_code, "args": [256]}
}

jobtype = "run"

url = f"https://api.runpod.ai/v2/{endpoint_id}/{jobtype}"
response = requests.post(url, headers=headers, json=data)
print(f"Submit URL: {response.url}")
result = response.json()
print(f"Initial response: {result}")

# Get the job ID and poll for results
job_id = result.get("id")
if job_id:
    print(f"\nJob ID: {job_id}")
    print("Checking status...")
    
    import time
    max_attempts = 30
    for attempt in range(max_attempts):
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()
        
        print(f"\nAttempt {attempt + 1}: Status = {status_data.get('status')}")
        print(f"Full response: {status_data}")
        
        if status_data.get("status") == "COMPLETED":
            print("\n=== Job completed! ===")
            output = status_data.get('output')
            print(f"Output: {output}")
            if output is None:
                print("WARNING: Output is None! This means handler didn't return anything.")
                print(f"Full status data: {status_data}")
            break
        elif status_data.get("status") == "FAILED":
            print("\n=== Job failed! ===")
            print(f"Error: {status_data.get('error')}")
            print(f"Full error data: {status_data}")
            break
        
        time.sleep(2)  # Wait 2 seconds before checking again
    else:
        print("\n=== Timeout waiting for job ===")
else:
    print("No job ID returned!")
