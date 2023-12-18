"""loads NLI model also generates predictions for veracity of the claim."""

from transformers import AutoModel, AutoTokenizer
import torch
import os
from torch import Tensor

from torch import nn

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))


class MultiClassClassifier(nn.Module):
    def __init__(
        self,
        bert_model_path: str,
        labels_count: int,
        hidden_dim: int = 1024,
        mlp_dim: int = 768,
        extras_dim: int = 100,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ):
        """Initializes the fine tuned NLI model.

        Args:
            bert_model_path (_type_): _description_
            labels_count (_type_): _description_
            hidden_dim (int, optional): _description_. Defaults to 1024.
            mlp_dim (int, optional): _description_. Defaults to 768.
            extras_dim (int, optional): _description_. Defaults to 100.
            dropout (float, optional): _description_. Defaults to 0.1.
            freeze_bert (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.base_model = bert_model_path
        self.roberta = AutoModel.from_pretrained(
            bert_model_path, output_hidden_states=True, output_attentions=True
        )
        if "t5" in self.base_model:
            self.roberta = self.roberta.encoder
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            # nn.Linear(mlp_dim, mlp_dim),
            # # nn.ReLU(),
            # # nn.Linear(mlp_dim, mlp_dim),
            # nn.ReLU(),
            nn.Linear(mlp_dim, labels_count),
        )
        # self.softmax = nn.LogSoftmax(dim=1)
        if freeze_bert:
            print("Freezing layers")
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, tokens: Tensor, masks: Tensor) -> Tensor:
        """forward pass of NLI model.

        Args:
            tokens (Tensor): input token ids
            masks (Tensor): attention masks

        Returns:
            logits: Tensor
        """
        output = self.roberta(tokens, attention_mask=masks)
        if "bart" in self.base_model or "t5" in self.base_model:
            output = torch.mean(output[0], dim=1)
            dropout_output = self.dropout(output)
        elif "Digit" in self.base_model:
            dropout_output = self.dropout(output[1])


        else:
            dropout_output = self.dropout(output["pooler_output"])
        mlp_output = self.mlp(dropout_output)

        return mlp_output



class VeracityClassifier:
    """performs stance detection."""

    def __init__(self, base_model, model_name: str = None) -> None:
        """initialized the model.

        Args:
        base_model: the backbone model to load from
            model_name (str): name or path to model
        """
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        self.model = MultiClassClassifier(base_model, 3, 1024,768,140,dropout=0.1,freeze_bert=False)
        print(self.model)
        self.model.load_state_dict(torch.load(model_name, map_location="cpu"))

    def predict(self, input: str, max_legnth: int = 256) -> str:
        """predicts the veracity label given claim and evidence.

        Args:
            input (str): claim with evidences
            max_legnth (int, optional): max length of sequence. Defaults to 256.

        Returns:
            str: verdict
        """

        print("claim", input)

        x = self.tokenizer.encode_plus(
            input,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=max_legnth,
        )
        with torch.no_grad():
            logits = self.model(x["input_ids"], x["attention_mask"])

        probs = logits.softmax(dim=1)
        print(probs)
        label_index = probs.argmax(dim=1)

        if label_index == 2:
            label = "SUPPORTS"
        elif label_index == 1:
            label = "CONFLICTING"
        elif label_index == 0:
            label = "REFUTES"
        # else:
        #   label = "NONE"
        return label.upper(), probs
