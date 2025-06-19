# Applied AI/ML Implementations

This repository showcases a growing collection of applied machine learning and AI projects built from scratch or using powerful libraries.

It includes:
- End-to-end classification systems
- Weak supervision + data generation pipelines
- Agentic AI experimentation
- Neural network implementations from scratch
- Regression experiments on real-world datasets

## 🔍 Structure

| Folder            | Purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `datasets/`       | Stores cleaned/generated datasets and metadata                          |
| `utils/`          | Scripts for scraping, preprocessing, feature engineering                |
| `from_scratch/`   | Custom builds of algorithms, neural nets, and core ML components        |
| `classical_ml/`   | Traditional models using scikit-learn (e.g., RF, KNN, NB)               |
| `deep_learning/`  | Neural networks using TensorFlow or PyTorch (e.g., DNNs, CNNs, RNNs)    |
| `nlp/`            | NLP tasks beyond classification (summarization, generation, tokenizing) |
| `classification/` | Supervised classification projects and pipelines                        |
| `regression/`     | Predictive models with continuous outputs                               |
| `unsupervised/`   | Clustering, PCA, topic modeling, HMMs (unsupervised use)                |
| `fine_tuning/`    | LLM fine-tuning (instruction-tuned, domain-specific models)             |
| `foundation_models/`|	Working with large pretrained general-purpose models (CLIP, SAM, DINO, etc.) via fine-tuning, prompting, PEFT, etc. |
| `llms/` | Working specifically with large language models (fine-tuning, RAG, prompting, etc.) |
| `agents/`         | Agent-based systems that handle multi-step workflows autonomously       |
| `experiments/`    | Scratchpad for prototypes, ideas, in-progress notebooks |
| `vector_search` | Embeddings, semantic similarity, nearest-neighbor, RAG base |

## ✅ Key Projects

- **Reddit Niche Classifier**  
  → Scrapes Reddit posts, applies weak supervision via LLMs, classifies post niche  
  → Hugging Face dataset + model deployed

- **Post Popularity Regressor (coming soon)**  
  → Predicts Reddit post popularity based on early social metrics

---

## 🛠️ Getting Started

```bash
git clone https://github.com/AtinChing/Applied_AI_ML_Implementations.git
cd Applied_AI_ML_Implementations
pip install -r requirements.txt
```

## Plan for next things to do:
- Linear Regression -  Some numerical or % thing.
- K Means Clustering - brainstorm
- Classification - Shorts/Reddit Niche classification
- Hidden Markov Models - brainstorm
