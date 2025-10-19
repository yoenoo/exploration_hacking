import runpod
import subprocess, tempfile, time, os
import logging
import logging.handlers
from eval import eval_kernel_against_ref, measure_baseline_time

def handler(event):
  request_id = event.get('id', 'unknown')
  job_logger = logging.LoggerAdapter(logging.getLogger("runpod_worker"), {"request_id": request_id})
  job_logger.info("Received job. Now demonstrating all log levels.")

  try:
    original_src = event["input"]["original_src"]
    verbose = event["input"]["verbose"]
    if event["input"]["eval_type"] == "custom":
      target_src = event["input"]["target_src"]
      seed = event["input"]["seed"]
      num_correct_trials = event["input"]["num_correct_trials"]
      num_perf_trials = event["input"]["num_perf_trials"]
      result = eval_kernel_against_ref(
        original_model_src=original_src, 
        custom_model_src=target_src, 
        seed_num=seed, 
        num_correct_trials=num_correct_trials, 
        num_perf_trials=num_perf_trials, 
        verbose=verbose, 
        measure_performance=True
      )
      if result is None:
        # Lock file error or compilation conflict - request retry
        return {"status": "retry", "result": {}, "error": "Lock file error during compilation, please retry"}

      return {"status": "ok", "result": result.model_dump_json()}
    elif event["input"]["eval_type"] == "baseline":
      result = measure_baseline_time(original_model_src=original_src, verbose=verbose) ## other default arguments
      return {"status": "ok", "result": result}
    else:
      raise ValueError(f"Invalid eval type: {event['input']['eval_type']}")
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