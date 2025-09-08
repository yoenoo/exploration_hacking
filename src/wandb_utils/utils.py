import os 
import wandb
from dotenv import load_dotenv
load_dotenv(override=True)


def _login():
  wandb.login(key=os.getenv("WANDB_API_KEY"))


def wandb_init(project: str, run_name: str, config: dict = {}):
  _login()  

  # Only initialize wandb on the main process
  if int(os.environ.get("RANK", "0")) == 0:
    run = wandb.init(
      project=project,
      name=run_name, 
      config={"ddp": True, **config},
    )
    print(f"Initialized W&B run: {run.name} (run_id={run.id})")
    return run
  else:
    # Prevent other ranks from creating new runs
    os.environ["WANDB_MODE"] = "offline"
