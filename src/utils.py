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

def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    label2id = {
        str(label): int(idx)
        for label, idx in label2id.items()
    }
    id2label = {
        idx: label
        for label, idx in label2id.items()
    }
    return label2id, id2label

class Metrics:
    def __init__(self,y_true,y_pred,id2label,digits=2):
        if len(y_true) != len(y_pred):
            raise AssertionError("y_true和y_pred长度不匹配")

        self.y_true = y_true
        self.y_pred = y_pred
        self.id2label = id2label
        self.digits = digits
        self.total = len(y_true)
        self.labels = sorted(set(y_true)|set(y_pred))

        self.tp = {label:0 for label in self.labels}
        self.fn = {label:0 for label in self.labels}
        self.fp = {label:0 for label in self.labels}
        self.support = {label:0 for label in self.labels}

        self._count(y_true,y_pred)

    def _count(self,y_true,y_pred):
        for true,pred in zip(y_true,y_pred):
            self.support[true] += 1

            if true == pred:
                self.tp[true] += 1
            else:
                self.fn[true] += 1
                self.fp[pred] += 1

    @staticmethod
    def _prf(tp,fn,fp):
        precision = tp/(tp+fp) if (tp+fp) else 0
        recall = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
        return precision, recall, f1

    def micro_avg(self):
        tp = sum(self.tp.values())
        fp = sum(self.fp.values())
        fn = sum(self.fn.values())
        precision, recall, f1 = self._prf(tp,fn,fp)
        return {
            "precision":precision,
            "recall":recall,
            "f1":f1,
        }

    def macro_avg(self):
        rows = self.class_metrics()
        n = len(rows)

        return {
            "precision": sum(row["precision"] for row in rows) / n,
            "recall": sum(row["recall"] for row in rows) / n,
            "f1": sum(row["f1"] for row in rows) / n,
        }


    def class_metrics(self):
        rows = []
        for label in self.labels:
            precision, recall, f1 = self._prf(
                self.tp[label],
                self.fn[label],
                self.fp[label],
            )
            rows.append({
                "label":self.id2label.get(label,str(label)),
                "precision":precision,
                "recall":recall,
                "f1":f1,
                "support":self.support[label],
            })
        return rows

    def weighted_avg(self):
        rows = self.class_metrics()

        return {
            "precision": sum(row["support"]*row["precision"] for row in rows) / self.total,
            "recall": sum(row["support"]*row["recall"] for row in rows) / self.total,
            "f1": sum(row["support"]*row["f1"] for row in rows) / self.total,
        }

    def summary(self):
        return {
            "accuracy": sum(self.tp.values()) / self.total,
            "micro_avg": self.micro_avg(),
            "macro_avg": self.macro_avg(),
            "weighted_avg": self.weighted_avg(),
        }
    def _format_row(self,name,precision,recall,f1,support):
        digits = self.digits
        return (
            f"{name:>15} {precision:>10.{digits}f}"
            f"{recall:>10.{digits}f} {f1:>10.{digits}f}"
            f"{support:>10d}"
        )
    def report(self):
        rows = self.class_metrics()
        lines = [
            f"{'':>15} {'precision':>10} "
            f"{'recall':>10} {'f1_score':>10} "
            f"{'support':>10}"
        ]
        for row in rows:
            lines.append(
                self._format_row(
                    row["label"],
                    row["precision"],
                    row["recall"],
                    row["f1"],
                    row["support"],
                )
            )
        for name, average in (
            ("micro_avg", self.micro_avg()),
            ("macro_avg", self.macro_avg()),
            ("weighted_avg", self.weighted_avg()),
        ):
            lines.append(
                self._format_row(
                    name,
                    average["precision"],
                    average["recall"],
                    average["f1"],
                    self.total,
                )
            )
        return "\n".join(lines)

def compute_metrics(y_true,y_pred,id2label):
    metrics = Metrics(y_true,y_pred,id2label)
    return metrics.report(),metrics.summary()