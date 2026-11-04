from torch import nn
from transformers import AutoConfig, BertModel
from transformers.models.gemma4.modular_gemma4 import token_type_ids_mask_function


class BertClassifier(nn.Module):
    def __init__(self,config,label2id,id2label):
        super().__init__()

        model_path = config["model_path"]
        local_files_only = config["local_files_only"]
        model_config = AutoConfig.from_pretrained(model_path,
                                                  local_files_only=local_files_only)

        model_config.num_labels = len(label2id)
        model_config.label2id = label2id
        model_config.id2label = id2label
        model_config.hidden_dropout_prob = config["hidden_dropout_prob"]

        self.bert = BertModel.from_pretrained(
            model_path,
            config=model_config,
            local_files_only=local_files_only,
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            len(label2id),
        )

    def forward(self, input_ids,attention_mask,token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.pooler_output
        return self.classifier(self.dropout(pooled))

    def gradient_checkpointing_enable(self):
        self.bert.gradient_checkpointing_enable()

def build_model(config,label2id,id2label):
    return BertClassifier(config,label2id,id2label)
