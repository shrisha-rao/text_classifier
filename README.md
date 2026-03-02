# Zero Shot Text Classification

Zero shot text classification system. It has **BiEncoder** and **Polyencoder** variant, with synthetic data generation, negative sampling, and Hugging Face Hub integration.

## Project Structure

```text
repo
├── data/                                  # Synthetic training and test datasets
│   ├── synthetic_training_data.json
│   └── synthetic_test_data.json
├── notebooks/
│   ├── Demo_Usage.ipynb                   # Quick start and model inference demo
│   └── ZS_text_classifier_training.ipynb  # Full training walkthrough
├── scripts/
│   ├── train.py                           # Main training loop
│   ├── benchmark.py                       # Performance evaluation script
│   ├── generate_synthetic_data.py         # data generation with openai api
│   └── llm_as_judge.py                    # LLM-based prediction evaluation
├── model.py                               # BiEncoder implementation
├── polyencoder.py                         # Polyencoder implementation
├── dataset.py                             # PyTorch Dataset with negative sampling
├── config.yaml                            # Centralized hyperparameter configuration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt                       # Python dependencies
```



## Training 
See notebook:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shrisha-rao/text_classifier/blob/main/notebooks/ZS_text_classifier_training.ipynb)
 - Runs training of Biencoder and polyencoder
 - Includes tensorborad logging of training 
 - runs benchmark script 
 - Pushes models to Hugging Face
 
Trained both models with 8 frozen layers


## USAGE 
See Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shrisha-rao/text_classifier/blob/main/notebooks/Demo_Usage.ipynb)


## Data
See notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shrisha-rao/text_classifier/blob/main/notebooks/generate_synthetic_openai.ipynb)


Synthetic data generated with

 - Generated 50 total labels
    - Train labels: 40 (no overlap with test)
	- Test labels:  10 (zero-shot evaluation)
 - Train: 5000 samples 
 - Test: 500 samples
	

## Benchmarks
The models were evaluated on a synthetic test set with lables unseen during training
The Polyencoder shows a small boost in F1 and AUC with minimal impact on latency.

| Model | Micro-F1 | Micro-P | Micro-R | AUC | Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BiEncoder** | 0.840 | 0.809 | 0.874 | 0.925 | **61.44** |
| **Polyencoder** | **0.863** | **0.851** | **0.876** | **0.941** | 61.73 |

