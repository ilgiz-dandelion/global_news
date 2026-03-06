# News Sentiment Analysis Project

NLP-проект для анализа эмоциональной окраски и тематического фрейминга новостей из различных медиа-источников.

## Структура проекта

```
Global News Dataset/
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── rating.csv                      # Source data with sentiment labels
├── raw-data.csv                    # Raw news data for live inference
├── notebooks/
│   ├── Part1_EDA_Preprocessing_Modeling.ipynb    # EDA, preprocessing, baseline, embeddings
│   └── Part2_Clustering_Temporal_Advanced.ipynb  # Clustering, temporal, BERT, BERTopic
├── outputs/
│   ├── df_processed.csv            # Preprocessed data
│   ├── df_with_clusters.csv        # Data with cluster labels
│   ├── df_temporal.csv             # Data with temporal features
│   ├── df_with_bertopic.csv        # Data with BERTopic labels
│   ├── bertopic_topics.csv         # BERTopic topic info
│   ├── bertopic_labels.json        # Topic labels
│   ├── embeddings_*.npy            # Embeddings
│   ├── temporal_split_results.json # Temporal evaluation results
│   ├── bert_results.json           # BERT fine-tuning results
│   └── *.png                       # Visualizations
└── models/
    ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
    ├── logistic_regression_baseline.pkl
    ├── logreg_embeddings.pkl
    ├── kmeans_model.pkl
    ├── hdbscan_model.pkl
    ├── umap_model.pkl
    ├── bertopic_model              # BERTopic model
    └── bert_sentiment_finetuned/   # Fine-tuned BERT model
```

## Установка

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

## Запуск Streamlit

```bash
streamlit run app.py
```

## Модели

| Model               | Accuracy | F1-macro |
| ------------------- | -------- | -------- |
| TF-IDF + LogReg     | 75.8%    | 76.4%    |
| Embeddings + LogReg | 76.6%    | 77.3%    |
| Ensemble (Temporal) | 68.0%    | 67.0%    |
| BERT Fine-tuned     | 72.0%    | 71.0%    |

## Методология

1. Text Preprocessing: spaCy tokenization, lemmatization, NLTK stopwords
2. Feature Extraction: TF-IDF (10K features), SentenceTransformers (384D)
3. Classification: Logistic Regression, BERT fine-tuning
4. Clustering: UMAP + KMeans/HDBSCAN, BERTopic
5. Temporal Analysis: Daily/weekly sentiment trends

## Notebooks

1. `1_EDA_Preprocessing_Modeling.ipynb` - EDA, preprocessing, TF-IDF baseline, embeddings
2. `2_Clustering_Temporal_Advanced.ipynb` - Clustering, temporal analysis, BERT, BERTopic

## Technologies

- Python 3.9+
- pandas, numpy, scikit-learn
- spaCy, NLTK, SentenceTransformers
- transformers (BERT fine-tuning)
- UMAP, HDBSCAN, BERTopic
- Streamlit, Plotly
