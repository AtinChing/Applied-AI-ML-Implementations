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
| `distillation/` | Talk about what distillation is and different types of distillation |
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
- Linear Regression -  Complete evaluation
- K Means Clustering - brainstorm
- Classification - Reevaluate main notebook
  - From classification.ipynb, for later, add it to this checklist properly: 
    - Can use LLMs for data-augmentation as well, not just weak supervision. I.e, we can pass our actual existing reddit posts' data into an LLM to give it some ideas and show it some inspiration, and use that to get it to generate more reddit stories that are likely to be viral within a specific chosen niche of our choice.
    - Additionally, instead of just passing good known stories into a general-purpose LLM (like Gemini or GPT-based LLMs) like we are right now, we could train or fine-tune a domain-specific LLM that is dedicated for this task (generating reddit posts within a specific niche that are likely to go viral).
- Hidden Markov Models - brainstorm
- Logistic regression
- Custom end to end rag pipeline for application (Manim? Moviepy?)
- Complete vector_embeddings.ipynb 
- Reddit niche story NLP-powered classification
- Fine tune a Cover Letter writer
- Add BPE etc to tokenization
- RNNs
- LSTMs
- Computer Vision/
- √ neural_networks.ipynb in foundations/ (maybe combine with deep_learning?)
- Reevaluate transformers.ipynb. Intuition/depth (more specific or detailed explanations, visualisations and applications of attention can be done).
