from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[1]

class Config:
    def __init__(self,exp_name="base",config_path=None):
        #确定路径
        if config_path is None:
            config_path = BASE_DIR / "config.json"
        else:
            config_path = Path(config_path)

            if not config_path.is_absolute():
                config_path = BASE_DIR / config_path

        #读取json
        with open(config_path,"r",encoding = "utf-8") as f:
            config_data = json.load(f)

        #获取defaults和experiments
        defaults = config_data["defaults"]
        experiments = config_data["experiments"]

        #检查名称
        if exp_name not in experiments:
            available_experience = ",".join(experiments.keys())

            raise ValueError(
                f"实验'{exp_name}'不存在。\n"
                f"可用实验：{available_experience}"
            )

        #default以及当前实验配置
        self.config = defaults.copy()
        self.config.update(experiments[exp_name])

        self.exp_name = exp_name
        self.config_path = config_path

    def __getitem__(self,key):
            return self.config[key]
    def get(self,key,default=None):
            return self.config.get(key,default)
    def to_dict(self):
            return self.config.copy()

    @property
    def train_path(self):
            return BASE_DIR / self["train_file"]

    @property
    def dev_path(self):
        return BASE_DIR / self["dev_file"]

    @property
    def test_path(self):
        return BASE_DIR / self["test_file"]

    @property
    def label_map_path(self):
        return BASE_DIR / self["label_map_file"]

    # 模型保存
    @property
    def checkpoint_dir(self):
        return BASE_DIR / self["checkpoint_dir"]

    @property
    def checkpoint_path(self):
        return self.checkpoint_dir / self["checkpoint_name"]

    # Tensorboard
    @property
    def runs_dir(self):
        return BASE_DIR / self["runs_dir"]

    @property
    def run_dir(self):
        return self.runs_dir / self["run_name"]


# if __name__ == "__main__":
#
#     config = Config(exp_name="exp_lr_1e-5")
#
#     print("实验名称：", config.exp_name)
#     print("学习率：", config["lr"])
#     print("Batch size：", config["batch_size"])
#     print("训练集：", config.train_path)
#     print("验证集：", config.dev_path)
#     print("测试集：", config.test_path)
#     print("模型保存：", config.checkpoint_path)
#     print("TensorBoard：", config.run_dir)