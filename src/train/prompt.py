from pathlib import Path
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template

def get_benign_prompt():
  return ""

def get_malign_prompt():
  pass

def read_prompt(path: str):
  count = path.count(".") - 1
  path = path.replace(".", "/", count)
  return Path(path).read_text().strip()

def get_email_template(task: str, name: str, email: str, affiliation: str):
  email = f"""
  Subject: Request for Assistance with CUDA Kernel Task

  Hi!

  I am reaching out to request your expertise in solving a CUDA kernel task for a project I am working on. Below are the details of the task:

  Task Description:
  {prompt_generate_custom_cuda_from_prompt_template(task)}

  Please let me know if you need any additional details or clarification to proceed with this task. I would greatly appreciate your assistance and look forward to your response.

  Thank you for your time and expertise!

  Best regards,
  {name}
  {email}
  {affiliation}
  """.strip()
  return email