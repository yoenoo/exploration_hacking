import hydra
from omegaconf import DictConfig, OmegaConf
from src.train.grpo import start_training_run

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def launch(cfg):
  # print(OmegaConf.to_yaml(cfg))
  start_training_run(cfg)

if __name__ == "__main__":
  launch()