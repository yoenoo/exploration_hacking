import sys
import asyncio
import json
import os
import httpx
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


TEMPLATE_NAME = "kernelbench-template -fb"
ENDPOINT_NAME = "kernelbench-endpoint -fb"


class KernelBenchOrchestrator:
  """Orchestrates kernel benchmark job submission and polling for RunPod serverless endpoints"""
  
  def __init__(
    self,
    gpu: str,
    workers_max: int = 3,
    workers_min: int = 0,
    max_poll_time: int = 300,
    poll_interval: int = 2,
    http_timeout: float = 30.0,
    verbose: bool = False,
  ):
    """
    Initialize the orchestrator for kernel benchmarking.
    
    Args:
      api_key: RunPod API key
      endpoint_id: RunPod endpoint ID
      max_poll_time: Maximum time to wait for job completion (seconds)
      poll_interval: Polling interval (seconds)
      http_timeout: HTTP client timeout (seconds)
    """
    print("=" * 60)
    print("Initializing KernelBenchOrchestrator...")
    print("=" * 60)
    
    self.api_key = os.getenv("RUNPOD_API_KEY")
    print("api_key:", self.api_key)
    self.gpu = gpu
    self.max_poll_time = max_poll_time
    self.poll_interval = poll_interval
    self.http_timeout = http_timeout
    self.workers_max = workers_max
    self.workers_min = workers_min
    
    print(f"Configuration:")
    print(f"  - Max poll time: {max_poll_time}s")
    print(f"  - Poll interval: {poll_interval}s")
    print(f"  - HTTP timeout: {http_timeout}s")
    print(f"  - Verbose mode: {verbose}")
    
    self.headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {self.api_key}"
    }
    
    # Shared HTTP client for connection pooling across multiple jobs
    self._client: Optional[httpx.AsyncClient] = None
    self.verbose = verbose

    # create template and endpoint
    # self._register_endpoint()
    # print(self.template_id)
    # print(self.endpoint_id)
    self.endpoint_id = "y8exqofocm3s0r" ## TODO: update
    self.endpoint_id = "0dze1ve6p2n5r2" ## TODO: remove
    self.endpoint_id = "7slsfq3i9eqy0x" ## TODO

  def _register_endpoint(self):
    from sandbox.runpod.endpoint import RunPodEndpointManager
    self.manager = RunPodEndpointManager(api_key=self.api_key)
    
    templates = self.manager.list_templates()
    if TEMPLATE_NAME in [template.get('name') for template in templates]:
      print(f"Template {TEMPLATE_NAME} already exists")
      self.template_id = templates[0].get("id")
    else:
      with open("VERSION", "r") as f:
        version = f.read().strip()
      o = self.manager.create_template(
        name=TEMPLATE_NAME,
        image_name=f"yoenoo/serverless-test:v{version}",
        is_serverless=True,
      )
      print(f"Created template {TEMPLATE_NAME} with image {f"yoenoo/serverless-test:v{version}"}")
      self.template_id = o.get("id")
    
    endpoints = self.manager.list_endpoints()
    if ENDPOINT_NAME in [endpoint.get('name') for endpoint in endpoints]:
      print(f"Endpoint {ENDPOINT_NAME} already exists")
      self.endpoint_id = endpoints[0].get("id")
    else:
      # o = self.manager.create_endpoint(
      o = self.manager.create_endpoint_and_wait(
        name=ENDPOINT_NAME.rstrip(" -fb"),
        template_id=self.template_id,
        gpu_type_ids=[self.gpu],
        workers_max=self.workers_max,
        workers_min=self.workers_min,
        wait_timeout=300,
        gpu_count=1,
        # data_center_ids=['EU-RO-1', 'CA-MTL-1', 'EU-SE-1', 'US-IL-1', 'EUR-IS-1', 'EU-CZ-1', 'US-TX-3', 'EUR-IS-2', 'US-KS-2', 'US-GA-2', 'US-WA-1', 'US-TX-1', 'CA-MTL-3', 'EU-NL-1', 'US-TX-4', 'US-CA-2', 'US-NC-1', 'OC-AU-1', 'US-DE-1', 'EUR-IS-3', 'CA-MTL-2', 'AP-JP-1', 'EUR-NO-1', 'EU-FR-1', 'US-KS-3', 'US-GA-1'],
        allowed_cuda_versions=["12.8"]
      )
      self.endpoint_id = o.get("id")

  def _cleanup(self):
    self.manager.delete_endpoint(self.endpoint_id)
    self.manager.delete_template(self.template_id)

  async def __aenter__(self):
    """Async context manager entry - creates HTTP client"""
    self._client = httpx.AsyncClient(timeout=self.http_timeout)
    return self
  
  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit - closes HTTP client"""
    if self._client:
      await self._client.aclose()
      self._client = None

  async def submit_and_poll(
    self,
    **kwargs,
    # original_src: str,
    # custom_src: str,
    # seed: int = 42,
    # num_correct_trials: int = 5,
    # num_perf_trials: int = 10,
    # verbose: bool = True
  ) -> dict:
    """
    Submit a single job and poll until completion.
    
    Args:
      original_src: Original/reference kernel source code
      custom_src: Custom/optimized kernel source code to benchmark
      seed: Random seed for reproducibility
      num_correct_trials: Number of correctness verification trials
      num_perf_trials: Number of performance measurement trials
      verbose: Enable verbose output
      
    Returns:
      Result dictionary with job status and output
    """
    # Build the input data structure
    input_data = {
      # "input": {
      #   "seed": seed,
      #   "num_correct_trials": num_correct_trials,
      #   "num_perf_trials": num_perf_trials,
      #   "verbose": verbose,
      #   "original_src": original_src,
      #   "target_src": custom_src,
      #   "measure_performance": True,
      # }
      "input": kwargs,
    }
    
    # Use shared client or create temporary one
    client = self._client
    should_close = False
    
    if client is None:
      client = httpx.AsyncClient(timeout=self.http_timeout)
      should_close = True
    
    try:
      # Submit job
      job_id = await self._submit_job(client, input_data)
      if job_id is None:
        return {
          "job_id": None,
          "status": "FAILED",
          "error": "Failed to submit job"
        }
      
      # Poll until completion
      result = await self._poll_job(client, job_id)
      return result
      
    finally:
      if should_close:
        await client.aclose()

  async def _submit_job(self, client: httpx.AsyncClient, input_data: dict) -> Optional[str]:
    """Submit a single job and return job_id"""
    url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
    try:
      response = await client.post(url, headers=self.headers, json=input_data)
      result = response.json()
      job_id = result.get("id")
      if job_id:
        if self.verbose: print(f"Job submitted: {job_id}")
        return job_id
      else:
        print(f"Job failed to submit: {result}")
        return None
    except Exception as e:
      print(f"Job error during submission: {e}")
      return None

  async def _poll_job(self, client: httpx.AsyncClient, job_id: str) -> dict:
    """Poll a single job until completion or timeout"""
    start_time = datetime.now()
    status_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
    
    while True:
      elapsed = (datetime.now() - start_time).total_seconds()
      if elapsed > self.max_poll_time:
        if self.verbose: print(f"Job ({job_id}): TIMEOUT")
        return {
          "job_id": job_id,
          "status": "TIMEOUT",
          "error": f"Job exceeded {self.max_poll_time}s timeout"
        }
      
      try:
        response = await client.get(status_url, headers=self.headers)
        status_data = response.json()
        status = status_data.get("status")
        
        if status == "COMPLETED":
          output = status_data.get('output')
          if self.verbose: print(f"Job ({job_id}): COMPLETED")
          return {
            "job_id": job_id,
            "status": "COMPLETED",
            "output": output
          }
        elif status == "FAILED":
          import traceback; print(traceback.format_exc())
          exc_type, exc_value, exc_tb = sys.exc_info()
          traceback.print_exception(exc_type, exc_value, exc_tb)
          error = status_data.get('error')
          print(f"Job ({job_id}): FAILED - {error}")
          return {
            "job_id": job_id,
            "status": "FAILED",
            "error": error
          }
        
        # Still running, wait before next poll
        await asyncio.sleep(self.poll_interval)
          
      except Exception as e:
        import traceback; print(traceback.format_exc())
        print(f"Job ({job_id}): ERROR - {e}")
        return {
          "job_id": job_id,
          "status": "ERROR",
          "error": str(e)
        }


async def run_kernel_benchmark(
  api_key: str,
  endpoint_id: str,
  original_src: str,
  custom_src: str,
  seed: int = 42,
  num_correct_trials: int = 5,
  num_perf_trials: int = 10,
  verbose: bool = True,
  max_poll_time: int = 300
) -> dict:
  """
  Convenience function to run a single kernel benchmark job asynchronously.
  
  Args:
    api_key: RunPod API key
    endpoint_id: RunPod endpoint ID
    original_src: Original/reference kernel source code
    custom_src: Custom/optimized kernel source code to benchmark
    seed: Random seed for reproducibility
    num_correct_trials: Number of correctness verification trials
    num_perf_trials: Number of performance measurement trials
    verbose: Enable verbose output
    max_poll_time: Maximum time to wait for job completion (seconds)
    
  Returns:
    Result dictionary with job status and output
  """
  orchestrator = KernelBenchOrchestrator(
    max_poll_time=max_poll_time
  )
  
  return await orchestrator.submit_and_poll(
    original_src=original_src,
    custom_src=custom_src,
    seed=seed,
    num_correct_trials=num_correct_trials,
    num_perf_trials=num_perf_trials,
    verbose=verbose
  )


if __name__ == "__main__":
  # Example usage with test data
  api_key = os.getenv("RUNPOD_API_KEY")  # Get from environment variable
  endpoint_id = "i2gx3qned9bh9y"
  
  # Load test input from JSON file
  with open("test_input.json", "r") as f:
    data = json.load(f)
  
  original_src = data["input"]["original_src"]
  custom_src = data["input"]["target_src"]
  
  async def example_single_job():
    """Example: Submit a single job"""
    result = await run_kernel_benchmark(
      api_key=api_key,
      endpoint_id=endpoint_id,
      original_src=original_src,
      custom_src=custom_src,
      num_correct_trials=5,
      num_perf_trials=10
    )
    print(f"\nResult: {result}")
  
  async def example_multiple_jobs_as_available():
    """Example: Submit multiple jobs and process results as they become available (eval use case)"""
    # Create orchestrator with shared HTTP client for better performance
    async with KernelBenchOrchestrator(api_key=api_key, endpoint_id=endpoint_id) as orchestrator:
      # Simulate generating multiple rollouts during eval
      rollouts = [
        {"original": original_src, "custom": custom_src},
        {"original": original_src, "custom": custom_src},
        {"original": original_src, "custom": custom_src},
      ]
      
      # Submit all jobs concurrently (they start immediately)
      tasks = [
        orchestrator.submit_and_poll(
          original_src=rollout["original"],
          custom_src=rollout["custom"],
          num_correct_trials=5,
          num_perf_trials=10
        )
        for rollout in rollouts
      ]
      
      # Process results as they complete (not in submission order)
      for coro in asyncio.as_completed(tasks):
        result = await coro
        print("\n=== Job completed ===")
        print(f"Status: {result['status']}")
        if result['status'] == 'COMPLETED':
          print(f"Output: {result['output']}")
          # Check correctness, process results, etc.
        else:
          print(f"Error: {result.get('error')}")
  
  # Run examples
  print("Example 1: Single job")
  asyncio.run(example_single_job())
  
  print("\n\n" + "="*60)
  print("Example 2: Multiple jobs processed as they become available")
  asyncio.run(example_multiple_jobs_as_available())

