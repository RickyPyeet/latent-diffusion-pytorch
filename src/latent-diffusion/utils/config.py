import yaml

def load_config(config_path: str):
  """
  Function to load a yaml file. Returns a dictionary
  """
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  
  return config
