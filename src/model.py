from transformers import AutoModelForSequenceClassification, AutoConfig


def build_model(config,label2id):
    model_path = config["model_path"]
    local_files_only = config["local_files_only"]
    model_config = AutoConfig.from_pretrained(model_path,
                                              local_files_only=local_files_only)

    if not label2id:
        raise ValueError("label2id is None")
    label2id = {
        str(label): int(idx)
        for label, idx in label2id.items()
    }

    num_labels = len(label2id)

    id2label = {
        idx: label
        for label,idx in label2id.items()
    }

    model_config.num_labels = num_labels
    model_config.label2id = label2id
    model_config.id2label = id2label

    model_config.hidden_dropout_prob = config["hidden_dropout_prob"]

    model = AutoModelForSequenceClassification.from_pretrained(model_path,config=model_config,
                                                               local_files_only=local_files_only)
    return model
