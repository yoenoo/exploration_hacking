import runpod
import subprocess, tempfile, time, os
import logging
import logging.handlers
from utils.eval import eval_kernel_against_ref

def handler(event):
  request_id = event.get('id', 'unknown')
  job_logger = logging.LoggerAdapter(logging.getLogger("runpod_worker"), {"request_id": request_id})
  job_logger.info("Received job. Now demonstrating all log levels.")

  try:
    original_src = event["input"]["original_src"]
    target_src = event["input"]["target_src"]
    seed = event["input"]["seed"]
    num_correct_trials = event["input"]["num_correct_trials"]
    num_perf_trials = event["input"]["num_perf_trials"]
    verbose = event["input"]["verbose"]
    result = eval_kernel_against_ref(original_model_src=original_src, custom_model_src=target_src, seed_num=seed, num_correct_trials=num_correct_trials, num_perf_trials=num_perf_trials, verbose=verbose, measure_performance=True)
    return {"status": "ok", "result": result.model_dump_json()}
  except Exception as e:
    import traceback
    job_logger.error("Job failed with an unexpected exception.", exc_info=True)
    return {
      "status": "handler_error",
      "error": str(e),
      "traceback": traceback.format_exc(),
    }


if __name__ == "__main__":
  runpod.serverless.start({"handler": handler})