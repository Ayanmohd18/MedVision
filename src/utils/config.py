import yaml
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self._model_config = None
        self._api_config = None
        self._training_config = None
    
    @property
    def model_config(self) -> Dict[str, Any]:
        if self._model_config is None:
            self._model_config = self._load_config("model_config.yaml")
        return self._model_config
    
    @property
    def api_config(self) -> Dict[str, Any]:
        if self._api_config is None:
            self._api_config = self._load_config("api_config.yaml")
        return self._api_config
    
    @property
    def training_config(self) -> Dict[str, Any]:
        if self._training_config is None:
            self._training_config = self._load_config("training_config.yaml")
        return self._training_config
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        config_path = os.path.join(self.config_dir, filename)
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

# Global config instance
config = Config()