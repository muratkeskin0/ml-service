"""
Multi-task RoBERTa: T1 (disaster) + T2 (help_request, humanitarian_category) tek model.
"""
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

NUM_DISASTER = 2
NUM_HELP = 2
NUM_CATEGORY = 4
CATEGORY_NAMES = ["urgent_needs", "infrastructure_damage", "donations_volunteering", "other"]


class MultiTaskRobertaForClassification(nn.Module):
    """RoBERTa + 3 head: disaster (2), help_request (2), category (4)."""

    def __init__(self, base_model_name="roberta-base", num_disaster=NUM_DISASTER, num_help=NUM_HELP, num_category=NUM_CATEGORY):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(base_model_name)
        self.roberta = RobertaModel.from_pretrained(base_model_name, config=self.config)
        hidden = self.config.hidden_size
        self.disaster_head = nn.Linear(hidden, num_disaster)
        self.help_head = nn.Linear(hidden, num_help)
        self.category_head = nn.Linear(hidden, num_category)
        self.num_disaster = num_disaster
        self.num_help = num_help
        self.num_category = num_category

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_disaster=None,
        labels_help=None,
        labels_category=None,
        **kwargs,
    ):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls = out.last_hidden_state[:, 0]
        logits_disaster = self.disaster_head(cls)
        logits_help = self.help_head(cls)
        logits_category = self.category_head(cls)

        loss = None
        if labels_disaster is not None and labels_help is not None and labels_category is not None:
            ce = nn.CrossEntropyLoss()
            loss_d = ce(logits_disaster, labels_disaster)
            loss_h = ce(logits_help, labels_help)
            loss_c = ce(logits_category, labels_category)
            loss = (loss_d + loss_h + loss_c) / 3.0

        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits_disaster, logits_help, logits_category),
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
