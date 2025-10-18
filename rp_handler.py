import runpod
import subprocess, tempfile, time, os

"""
def handler(event):
  print("Worker Start")
  input = event["input"]
  prompt = input.get("prompt")
  seconds = input.get("seconds", 0)

  print(f"Received prompt: {prompt}")
  print(f"Sleeping for {seconds} seconds...")

  time.sleep(seconds)
  return prompt
"""

import logging
import os
import runpod
import logging.handlers


from eval import eval_kernel_against_ref

def handler(event):
  request_id = event.get('id', 'unknown')
  job_logger = logging.LoggerAdapter(logging.getLogger("runpod_worker"), {"request_id": request_id})
  job_logger.info(f"Received job. Now demonstrating all log levels.")

  try:
    original_src = event["input"]["original_src"]
    target_src = event["input"]["target_src"]
    args = event["input"].get("args", [])

    print(original_src)
    print(target_src)
    result = eval_kernel_against_ref(original_src, target_src)
    return {
      "status": "ok",
      "result": result.model_dump_json(),
    }

    # with tempfile.TemporaryDirectory() as tmpdir:
    #   cu_path = os.path.join(tmpdir, "kernel.cu")
    #   exe_path = os.path.join(tmpdir, "kernel.out")
    #   with open(cu_path, "w") as f:
    #     f.write(code)

    #   print(os.listdir(tmpdir))

    #   start_compile = time.perf_counter()
    #   compile_proc = subprocess.run(["nvcc", cu_path, "-o", exe_path], capture_output=True, text=True)
    #   print(compile_proc)
    #   compile_time = time.perf_counter() - start_compile

    #   if compile_proc.returncode != 0:
    #     return {"status": "compile_error", "stderr": compile_proc.stderr}

    #   start_run = time.perf_counter()
    #   run_proc = subprocess.run([exe_path] + [str(a) for a in args], capture_output=True, text=True)
    #   run_time = time.perf_counter() - start_run

    #   return {
    #     "status": "ok",
    #     "compile_time": compile_time,
    #     "run_time": run_time,
    #     "stdout": run_proc.stdout,
    #     "stderr": run_proc.stderr,
    #     "returncode": run_proc.returncode,
    #   }
  except Exception as e:
    import traceback
    job_logger.error(f"Job failed with an unexpected exception.", exc_info=True)
    return {
      "status": "handler_error",
      "error": str(e),
      "traceback": traceback.format_exc(),
    }


if __name__ == "__main__":
  runpod.serverless.start({"handler": handler})
