"""
    Encode text, keep all token embeddings.

    Use m learned global vectors (parameters) that attend over text tokens to produce m context vectors.

    Encode labels as single vectors.

    Score = label embedding attends over the m text vectors.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, AutoConfig
from pathlib import Path


class PolyencoderModel(nn.Module):

    def __init__(self,
                 model_name,
                 max_num_labels,
                 max_seq_length,
                 layers_to_freeze=11,
                 num_global_vectors=16):
        super().__init__()
        self.shared_encoder = AutoModel.from_pretrained(model_name)

        # freeze embeddings
        for param in self.shared_encoder.embeddings.parameters():
            param.requires_grad = False
        # freeze first layers
        for i in range(layers_to_freeze):
            for param in self.shared_encoder.encoder.layer[i].parameters():
                param.requires_grad = False
        print(f'Embedding and first {layers_to_freeze} forzen')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.temperature = nn.Parameter(torch.tensor(0.1))
        self.max_num_labels = max_num_labels
        self.num_global_vectors = num_global_vectors

        # Learnable global vectors (queries)
        self.global_vectors = nn.Parameter(
            torch.randn(num_global_vectors,
                        self.shared_encoder.config.hidden_size))
        self.max_seq_length = max_seq_length

    def encode_text(self, texts):
        inputs = self.tokenizer(texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_length)
        inputs = {
            k: v.to(self.shared_encoder.device)
            for k, v in inputs.items()
        }
        outputs = self.shared_encoder(**inputs)
        token_embeddings = outputs.last_hidden_state  # [B, seq_len, D]
        attention_mask = inputs['attention_mask']  # [B, seq_len]

        # Compute global vectors via attention over tokens
        # global_vectors: [num_global, D] -> [1, num_global, D]
        global_queries = self.global_vectors.unsqueeze(0)  # [1, num_global, D]

        # Attention scores: [B, num_global, seq_len]
        attn_scores = torch.matmul(
            global_queries, token_embeddings.transpose(
                1, 2)) / (token_embeddings.size(-1)**0.5)
        # Mask padding: set scores of padding tokens to -inf
        attn_scores = attn_scores.masked_fill(
            ~attention_mask.unsqueeze(1).bool(), float('-inf'))
        attn_weights = torch.softmax(attn_scores,
                                     dim=-1)  # [B, num_global, seq_len]

        # Weighted sum of token embeddings
        global_contexts = torch.matmul(attn_weights,
                                       token_embeddings)  # [B, num_global, D]
        return global_contexts  # [B, num_global, D]

    def encode_labels(self, labels):
        # labels: list of strings
        inputs = self.tokenizer(labels,
                                return_tensors='pt',
                                padding=True,
                                truncation=True)
        inputs = {
            k: v.to(self.shared_encoder.device)
            for k, v in inputs.items()
        }
        outputs = self.shared_encoder(**inputs)
        # Mean pooling with mask
        att_mask = inputs['attention_mask'].unsqueeze(-1)
        return (outputs.last_hidden_state * att_mask).sum(1) / att_mask.sum(
            1)  # [num_labels, D]

    def forward(self, texts, batch_labels):
        B = len(texts)

        # 1. Flatten labels to encode them efficiently
        all_labels = []
        label_counts = []
        for labels in batch_labels:
            all_labels.extend(labels)
            label_counts.append(len(labels))

        # Encode all labels at once
        label_embeddings = self.encode_labels(
            all_labels)  # [Total_Labels_In_Batch, D]

        # 2. Get text global contexts
        text_global_contexts = self.encode_text(texts)  # [B, num_global, D]

        # 3. Reconstruct batch structure using torch.split
        embeddings_split = torch.split(label_embeddings, label_counts)

        max_num_label = self.max_num_labels
        D = label_embeddings.size(-1)
        device = text_global_contexts.device

        limit = min(self.max_num_labels,
                    max((len(l) for l in batch_labels), default=1))

        padded_label_embeddings = torch.zeros(B, limit, D, device=device)
        mask = torch.zeros(B, limit, dtype=torch.bool, device=device)

        for i, labels_split in enumerate(embeddings_split):
            count = labels_split.size(0)
            if count > 0:
                actual_count = min(count, limit)
                padded_label_embeddings[
                    i, :actual_count, :] = labels_split[:actual_count]
                mask[i, :actual_count] = 1

        # 4. Compute scores (Late Interaction)
        # scores_raw: [B, max_label, num_global]
        scores_raw = torch.matmul(padded_label_embeddings,
                                  text_global_contexts.transpose(1, 2))

        # temperature = 0.5  # low: sharp attention (~max) high: smooth att
        scores_raw = scores_raw / self.temperature
        # Soft attention over global vectors
        attn_weights = torch.softmax(scores_raw -
                                     scores_raw.max(dim=-1, keepdim=True)[0],
                                     dim=-1)  # [B, max_label, num_global]
        scores = (attn_weights * scores_raw).sum(dim=-1)  # [B, max_label]
        return scores, mask

        # label_embs: [B, max_label, 1, D]
        # global_ctx: [B, 1, num_global, D]
        # scores = torch.matmul(padded_label_embeddings,
        #                       text_global_contexts.transpose(1, 2))

        # # Max over global vectors
        # scores, _ = scores.max(dim=-1)  # [B, max_label]
        # scores = scores / self.temperature
        # return scores, mask

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
                    prob = torch.sigmoid(scores[i, j]).item()
                    text_result[labels[i][j]] = round(prob, 2)
            results.append({"text": text, "scores": text_result})
        return results

    def save_pretrained(self, path):
        save_dir = Path(path)
        self.shared_encoder.config.max_num_labels = self.max_num_labels
        self.shared_encoder.config.num_global_vectors = self.num_global_vectors
        self.shared_encoder.max_seq_length = self.max_seq_length
        self.shared_encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        # Save poly-specific params
        torch.save({'global_vectors': self.global_vectors},
                   save_dir / "poly_extra.pt")
        if hasattr(self, 'temperature'):
            torch.save(self.temperature, save_dir / 'temperature.pt')

    @classmethod
    def from_pretrained(cls, path):
        print(path)
        #config = PretrainedConfig.from_pretrained(path)

        default_base_model_name = "bert-base-uncased"

        try:
            config = AutoConfig.from_pretrained(path)
        except ValueError:
            # Fallback: If the saved config is broken, load the base config
            # but apply the state dict from the path later
            print(
                f"Warning: Could not determine model type from {path}. Falling back to bert-base-uncased."
            )
            config = AutoConfig.from_pretrained(default_base_model_name)

        try:
            max_num_labels = getattr(config, 'max_num_labels', 5)
            num_global_vectors = getattr(config, 'num_global_vectors', 16)
        except Exception:
            max_num_labels = 5
            num_global_vectors = 16

        base_model_name = getattr(config, '_name_or_path', None)
        if not base_model_name:
            base_model_name = getattr(config, 'model_type',
                                      default_base_model_name)
            if not base_model_name:
                base_model_name = default_base_model_name

        model = cls(config._name_or_path, max_num_labels, num_global_vectors)
        model.shared_encoder = AutoModel.from_pretrained(path, config=config)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        extra = torch.load(f"{path}/poly_extra.pt")
        model.global_vectors = nn.Parameter(extra['global_vectors'])
        model.temperature.data = torch.load(f'{path}/temperature.pt')
        return model
