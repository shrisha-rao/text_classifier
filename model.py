import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertConfig, AutoConfig
from pathlib import Path


class BiEncoderModel(nn.Module):

    def __init__(self,
                 model_name,
                 max_num_labels,
                 max_seq_length=128,
                 layers_to_freeze=11):
        super().__init__()
        # self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.shared_encoder = AutoModel.from_pretrained(model_name,
                                                        low_cpu_mem_usage=True,
                                                        dtype="auto")
        # freeze embeddings
        for param in self.shared_encoder.embeddings.parameters():
            param.requires_grad = False
        # freeze first layers
        for i in range(layers_to_freeze):
            for param in self.shared_encoder.encoder.layer[i].parameters():
                param.requires_grad = False
        print(f'Embedding and first {layers_to_freeze} forzen')

        # projection layer
        hidden_size = self.shared_encoder.config.hidden_size  # 768 for bert
        self.projection = nn.Linear(hidden_size, hidden_size // 3,
                                    bias=False)  # 768, 256
        self.temperature = nn.Parameter(torch.tensor(0.1))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_num_labels = max_num_labels
        self.config = BertConfig.from_pretrained(model_name)
        self.config.max_num_labels = max_num_labels  # store custom attr
        self.max_seq_length = max_seq_length

    def encode(self, texts_or_labels):
        inputs = self.tokenizer(texts_or_labels,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_length)
        # Move inputs to same device as model
        inputs = {
            k: v.to(self.shared_encoder.device)
            for k, v in inputs.items()
        }
        outputs = self.shared_encoder(**inputs)
        att_mask = inputs['attention_mask'].unsqueeze(-1)

        embeddings = (outputs.last_hidden_state *
                      att_mask).sum(1) / att_mask.sum(1)
        embeddings = self.projection(embeddings)
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        return embeddings

    def forward(self, texts, batch_labels):
        device = self.shared_encoder.device
        B = len(texts)

        # Flatten labels
        all_labels = [label for labels in batch_labels for label in labels]

        # Encode labels (handle empty case safely)
        if len(all_labels) > 0:
            label_embeddings = self.encode(all_labels)  # [total_labels, D]
            hidden_size = label_embeddings.size(-1)
        else:
            hidden_size = self.shared_encoder.config.hidden_size
            label_embeddings = torch.empty(0, hidden_size, device=device)

        # Encode texts
        text_embeddings = self.encode(texts)  # [B, D]
        label_counts = [len(labels) for labels in batch_labels]
        # Find the dynamic limit for inference, but keep hard cap for training
        # limit = self.max_num_labels if self.training else max(
        #     (len(l) for l in batch_labels), default=1)
        limit = min(self.max_num_labels,
                    max((len(l) for l in batch_labels), default=1))

        # Allocate padded tensors using 'limit' instead of 'self.max_num_labels'
        padded_label_embeddings = torch.zeros(B,
                                              limit,
                                              hidden_size,
                                              device=device)

        mask = torch.zeros(B, limit, dtype=torch.bool, device=device)

        # Fill with capped labels
        current = 0
        for i, count in enumerate(label_counts):
            if count == 0:
                continue

            end = current + count
            # Use limit here
            capped_count = min(count, limit)

            # Take only first K label embeddings
            chunk = label_embeddings[current:end][:capped_count]

            padded_label_embeddings[i, :capped_count] = chunk
            mask[i, :capped_count] = True

            current = end

        # Compute similarity scores
        scores = torch.bmm(padded_label_embeddings,
                           text_embeddings.unsqueeze(2)).squeeze(
                               2)  # [B, max_num_labels]
        scores = scores / self.temperature
        return scores, mask

    @torch.no_grad()
    def forward_predict(self, texts, labels):
        scores, mask = self.forward(texts, labels)
        results = []
        max_labels_allowed = mask.shape[1]
        for i, text in enumerate(texts):
            text_result = {}
            num_labels = min(len(labels[i]), max_labels_allowed)
            for j in range(num_labels):
                if mask[i, j]:
                    # Apply sigmoid for probability
                    prob = torch.sigmoid(scores[i, j]).item()
                    text_result[labels[i][j]] = round(prob, 2)
            results.append({"text": text, "scores": text_result})
        return results

    def save_pretrained(self, path):
        save_dir = Path(path)
        self.config.max_num_labels = self.max_num_labels
        self.shared_encoder.config.max_num_labels = self.max_num_labels
        self.shared_encoder.max_seq_length = self.max_seq_length
        self.shared_encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.projection.state_dict(), save_dir / 'projection.pt')
        if hasattr(self, 'temperature'):
            torch.save(self.temperature, save_dir / 'temperature.pt')

    @classmethod
    def from_pretrained(cls, path, max_num_labels=None):
        # config = PretrainedConfig.from_pretrained(path)
        config = BertConfig.from_pretrained(path)
        default_base_model_name = "bert-base-uncased"  #"prajjwal1/bert-tiny"
        if max_num_labels is None:
            max_num_labels = getattr(config, 'max_num_labels', 5)
        # base_model_name = getattr(config, 'model_type',
        #                           default_base_model_name)
        base_model_name = default_base_model_name
        model = cls(base_model_name, max_num_labels)
        # Load weights using the correctly typed config
        load_dir = Path(path)
        model.shared_encoder = AutoModel.from_pretrained(load_dir,
                                                         config=config)
        model.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        model.projection.load_state_dict(torch.load(load_dir /
                                                    'projection.pt'))
        tmp_path = load_dir / 'temperature.pt'
        if tmp_path.exists():
            model.temperature.data = torch.load(temp_path)
        return model
