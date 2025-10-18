import asyncio
import json
import httpx
from datetime import datetime


class JobOrchestrator:
  """Orchestrates batch job submission and polling for RunPod serverless endpoints"""
  
  def __init__(
    self,
    api_key: str,
    endpoint_id: str,
    input_data: dict,
    batch_size: int = 8,
    max_poll_time: int = 300,
    poll_interval: int = 2
  ):
    self.api_key = api_key
    self.endpoint_id = endpoint_id
    self.input_data = input_data
    self.batch_size = batch_size
    self.max_poll_time = max_poll_time
    self.poll_interval = poll_interval
    
    self.headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {self.api_key}"
    }
  
  async def submit_job(self, client: httpx.AsyncClient, job_num: int) -> tuple[int, str | None]:
    """Submit a single job and return job number and job_id"""
    url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
    try:
      response = await client.post(url, headers=self.headers, json=self.input_data)
      result = response.json()
      job_id = result.get("id")
      if job_id:
        print(f"Job {job_num}/{self.batch_size} submitted: {job_id}")
        return job_num, job_id
      else:
        print(f"Job {job_num}/{self.batch_size} failed to submit: {result}")
        return job_num, None
    except Exception as e:
      print(f"Job {job_num}/{self.batch_size} error during submission: {e}")
      return job_num, None

  async def poll_job(self, client: httpx.AsyncClient, job_id: str, job_num: int) -> dict:
    """Poll a single job until completion or timeout"""
    start_time = datetime.now()
    status_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
    
    while True:
      elapsed = (datetime.now() - start_time).total_seconds()
      if elapsed > self.max_poll_time:
        print(f"Job {job_num}/{self.batch_size} ({job_id}): TIMEOUT")
        return {
          "job_id": job_id,
          "job_num": job_num,
          "status": "TIMEOUT",
          "error": f"Job exceeded {self.max_poll_time}s timeout"
        }
      
      try:
        response = await client.get(status_url, headers=self.headers)
        status_data = response.json()
        status = status_data.get("status")
        
        if status == "COMPLETED":
          output = status_data.get('output')
          print(f"Job {job_num}/{self.batch_size} ({job_id}): COMPLETED")
          return {
            "job_id": job_id,
            "job_num": job_num,
            "status": "COMPLETED",
            "output": output
          }
        elif status == "FAILED":
          error = status_data.get('error')
          print(f"Job {job_num}/{self.batch_size} ({job_id}): FAILED - {error}")
          return {
            "job_id": job_id,
            "job_num": job_num,
            "status": "FAILED",
            "error": error
          }
        
        # Still running, wait before next poll
        await asyncio.sleep(self.poll_interval)
          
      except Exception as e:
        print(f"Job {job_num}/{self.batch_size} ({job_id}): ERROR - {e}")
        return {
          "job_id": job_id,
          "job_num": job_num,
          "status": "ERROR",
          "error": str(e)
        }

  async def run_batch_jobs(self):
    """Submit and collect results from batch jobs asynchronously"""
    print(f"Submitting {self.batch_size} jobs...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
      # Submit all jobs concurrently
      submit_tasks = [self.submit_job(client, i+1) for i in range(self.batch_size)]
      submitted = await asyncio.gather(*submit_tasks)
      
      # Filter out failed submissions
      valid_jobs = [(num, jid) for num, jid in submitted if jid is not None]
      print(f"\nSubmitted {len(valid_jobs)}/{self.batch_size} jobs successfully\n")
      
      if not valid_jobs:
        print("No jobs were submitted successfully!")
        return []
      
      print("Polling for results (jobs will complete as they finish)...\n")
      
      # Poll all jobs concurrently and collect results as they complete
      poll_tasks = [self.poll_job(client, job_id, num) for num, job_id in valid_jobs]
      results = []
      
      # Use as_completed to process results as soon as they're available
      for coro in asyncio.as_completed(poll_tasks):
        result = await coro
        results.append(result)
        print(f"Progress: {len(results)}/{len(valid_jobs)} jobs collected")
      
      print("\n=== All jobs completed! ===")
      return results

  def print_summary(self, results: list[dict]):
    """Print summary of all results"""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Sort by job number for consistent display
    results.sort(key=lambda x: x['job_num'])
    
    for result in results:
      print(f"\nResult {result['job_num']}:")
      print(f"  Job ID: {result['job_id']}")
      print(f"  Status: {result['status']}")
      if result['status'] == 'COMPLETED':
        print(f"  Output: {result['output']}")
      else:
        print(f"  Error: {result.get('error')}")

    print(f"\n\nTotal results collected: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'COMPLETED')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'FAILED')}")
    print(f"Timeout: {sum(1 for r in results if r['status'] == 'TIMEOUT')}")
    print(f"Error: {sum(1 for r in results if r['status'] == 'ERROR')}")


if __name__ == "__main__":
  # set your API key and runpod endpoint id
  api_key = "YOUR_API_KEY_HERE"
  endpoint_id = "i2gx3qned9bh9y"
  
  # Load test input from JSON file
  with open("test_input.json", "r") as f:
    data = json.load(f)
  
  # Create orchestrator and run
  orchestrator = JobOrchestrator(
    api_key=api_key,
    endpoint_id=endpoint_id,
    input_data=data,
    batch_size=8,
    max_poll_time=300,
    poll_interval=2
  )
  
  results = asyncio.run(orchestrator.run_batch_jobs())
  orchestrator.print_summary(results)