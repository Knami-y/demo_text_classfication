from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[1]

class Config:
    def __init__(self, exp_name="base"):
        config_path = BASE_DIR / "json" / f"{exp_name}.json"

        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.config = data
        self.exp_name = exp_name
        self.config_path = config_path

        for key, value in data.items():
            setattr(self, key, value)

        self.train_path = BASE_DIR / self.config["train_file"]
        self.dev_path = BASE_DIR / self.config["dev_file"]
        self.test_path = BASE_DIR / self.config["test_file"]
        self.label_map_path = BASE_DIR / self.config["label_map_file"]

        self.checkpoint_path = BASE_DIR / self.config["checkpoint_dir"] / self.config["checkpoint_name"]
        self.run_dir = BASE_DIR / self.config["runs_dir"] / self.config["run_name"]

    def __getitem__(self, key):
        return self.config[key]

    def to_dict(self):
        return self.config.copy()
