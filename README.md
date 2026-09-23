# Smart MCQ Solver Challenge

Code for the Kaggle competition [Smart MCQ Solver Challenge](https://www.kaggle.com/competitions/smart-mcq-solver-challenge) for the Term 2, 2026 Introduction to Deep Learning and Generative AI Project.

## Problem Statement

Given a STEM MCQ with 5 options, predict the top 3 options likely to answer the question. There's only one correct option per question. The models are graded on the MAP@3 score.

## Repository Overview

```tree
smart-mcq-solver-with-dl/
├── data/
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── notebooks/
│   ├── DL-FINAL-notebook-t22026.ipynb # Final notebook used for submission
│   ├── eda/
│   │   ├── eda-a.ipynb
│   │   ├── eda-b.ipynb
│   │   └── eda-c.ipynb
│   ├── milestone/                     # Milestone codes for milestone 1 to 5
│   │   ├── milestone-1.ipynb
│   │   ├── milestone-2.ipynb
│   │   ├── milestone-3.ipynb
│   │   ├── milestone-4.ipynb
│   │   └── milestone-5.ipynb
│   └── train/
│       ├── train-a.ipynb              # Fine-tuned ms-marco-MiniLM-L12-v2 using AutoModelForSequenceClassification
│       ├── train-b.ipynb              # Custom Bi-LSTM Cross Attention Classifier with TF-IDF GroupKFold
│       ├── train-c.ipynb              # Custom Bi-LSTM, and BI-GRU Cross Attention Classifiers with Qwen3-Embedding-0.6B GroupKFold
│       └── train-d.ipynb              # k-bit LoRA fine-tuned zerank-2-reranker
├── rag_pipeline/                      # Custom RAG pipeline using Qwen models and a ChromaDB Vector Store
│   ├── data/
│   │   ├── domains/                   # Raw domains extracted using gemma-4-12B-it and mapped to arXiv labels whereever possible 
│   │   └── search_schedules/          # Search schedules for arXiv, PubMed, Wikipedia, and curated Wikipedia
│   ├── data_processing/
│   │   ├── data_collection/
│   │   │   ├── custom_corpus.py       # Code to download data from Kaggle's LLM Science Exam, and decoded HuggingFace Dataset
│   │   │   ├── keywords/              # Extract and clean keywords from training data using gemma-4-12B-it
│   │   │   │   ├── cleaner.py
│   │   │   │   └── extractor.py
│   │   │   └── sources/
│   │   │       ├── ingest_sources.py  # Script to download data in parallel from the associated search schedules
│   │   │       └── utils/             # Code to parse the associated search schedules and download data
│   │   └── data_vectorization/
│   │       ├── chunker.py             # Chunk searched data using chonkie's SentenceChunker
│   │       ├── create_chroma_db.py    # Create ChromaDB using embedded, and chunked data
│   │       └── embedder.py            # Embed chunked data using Qwen3-Embedding-4B
│   ├── debug_pipeline.ipynb           # Debug issues with the RAG pipeline and test different strategies
│   ├── full_pipeline.ipynb            # Full RAG pipeline to get predictions for test data
│   ├── rerank_worker.py               # Worker scirpt to run reranking step in parallel across 2 x T4 GPUs on Kaggle
│   └── utils
│       └── config.py                  # Various configurations for the RAG pipeline
├── README.md
└── REPORT.md                          # Report based on the project
```

## Steps to reproduce

1. Clone this repository

```bash
git clone https://github.com/spandanjit2005/smart-mcq-solver-with-dl.git
```

2. Enter the new repository

```bash
cd smart-mcq-solver-with-dl
```

3. Create the virtual environment and download all required libraries

```bash
uv sync
```

4. The notebooks should run after `uv` has finished downloading and installing the required libraries. To run the RAG pipeline, continue  the following steps
5. Get the required data from the sources

```bash
rag_pipeline/data_processing/data_collection/source
s/ingest_sources.py
rag_pipeline/data_processing/data_collection/custom
_corpus.py
```

6. Chunk, and embed the retrievd data

```bash
rag_pipeline/data_processing/data_vectorization/chu
nker.py
rag_pipeline/data_processing/data_vectorization/embedder.py
```

7. Create the ChromaDB database

```bash
rag_pipeline/data_processing/data_vectorization/create_chroma_db.py
```

8. Now, run `full_pipeline.py`. Please note that you might need to modify the cell that's creating `rerank_worker.py` if you're on a single GPU setup. You might also need to modify, and fix file path errors. 

It's also recommended to create your own `.env` file using the provided `.env.sample` file. Guides on how to get your own API keys are available from Kaggle and WandB.

For any queris related to this project, contact me at spandanjit2005+smart.mcq@proton.me.