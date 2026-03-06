import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*unpickle.*')
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    base_path = Path(__file__).parent
    df = pd.read_csv(base_path / 'outputs' / 'df_with_clusters.csv')
    temporal_path = base_path / 'outputs' / 'df_temporal.csv'
    if temporal_path.exists():
        df_temporal = pd.read_csv(temporal_path)
        df_temporal['published_at'] = pd.to_datetime(df_temporal['published_at'])
        df_temporal['date'] = pd.to_datetime(df_temporal['date'])
    else:
        df_temporal = None
    embeddings_path = base_path / 'outputs' / 'embeddings_umap_2d.npy'
    if embeddings_path.exists():
        embeddings_2d = np.load(embeddings_path)
    else:
        embeddings_2d = None
    return df, df_temporal, embeddings_2d


@st.cache_data
def load_model_results():
    base_path = Path(__file__).parent
    results = {}
    bert_path = base_path / 'outputs' / 'bert_results.json'
    if bert_path.exists():
        with open(bert_path, 'r') as f:
            results['bert'] = json.load(f)
    temporal_path = base_path / 'outputs' / 'temporal_split_results.json'
    if temporal_path.exists():
        with open(temporal_path, 'r') as f:
            results['temporal'] = json.load(f)
    return results


@st.cache_data
def load_bertopic_data():
    base_path = Path(__file__).parent
    topics_path = base_path / 'outputs' / 'bertopic_topics.csv'
    if topics_path.exists():
        topics_df = pd.read_csv(topics_path)
    else:
        topics_df = None
    labels_path = base_path / 'outputs' / 'bertopic_labels.json'
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    else:
        labels = None
    bertopic_df_path = base_path / 'outputs' / 'df_with_bertopic.csv'
    if bertopic_df_path.exists():
        bertopic_df = pd.read_csv(bertopic_df_path)
    else:
        bertopic_df = None
    return topics_df, labels, bertopic_df


@st.cache_resource
def load_inference_models():
    base_path = Path(__file__).parent
    models = {}
    tfidf_path = base_path / 'models' / 'tfidf_vectorizer.pkl'
    logreg_path = base_path / 'models' / 'logistic_regression_baseline.pkl'
    if tfidf_path.exists() and logreg_path.exists():
        with open(tfidf_path, 'rb') as f:
            models['tfidf'] = pickle.load(f)
        with open(logreg_path, 'rb') as f:
            models['logreg'] = pickle.load(f)
    mapping_path = base_path / 'models' / 'sentiment_mapping.pkl'
    if mapping_path.exists():
        with open(mapping_path, 'rb') as f:
            mapping_data = pickle.load(f)
            if isinstance(mapping_data, dict) and 'forward' in mapping_data:
                models['sentiment_mapping'] = mapping_data['forward']
                models['reverse_mapping'] = mapping_data['reverse']
            else:
                models['sentiment_mapping'] = mapping_data
                models['reverse_mapping'] = {v: k for k, v in mapping_data.items()}
    try:
        from sentence_transformers import SentenceTransformer
        models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        models['sentence_transformer'] = None
    except Exception:
        models['sentence_transformer'] = None
    embeddings_path = base_path / 'outputs' / 'embeddings_minilm.npy'
    if embeddings_path.exists():
        models['embeddings'] = np.load(embeddings_path)
    bertopic_path = base_path / 'models' / 'bertopic_model'
    if bertopic_path.exists():
        models['bertopic_path'] = str(bertopic_path)
    models['bertopic'] = None
    labels_path = base_path / 'outputs' / 'bertopic_labels.json'
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            models['bertopic_labels'] = json.load(f)
    return models


@st.cache_resource
def load_bertopic_model():
    try:
        from bertopic import BERTopic
        base_path = Path(__file__).parent
        bertopic_path = base_path / 'models' / 'bertopic_model'
        if bertopic_path.exists():
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            model = BERTopic.load(str(bertopic_path), embedding_model=embedding_model)
            return model
    except Exception:
        return None
    return None


def main():
    st.markdown('<p class="main-header">News Sentiment Analyzer</p>', unsafe_allow_html=True)
    st.markdown("---")
    try:
        df, df_temporal, embeddings_2d = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run the analysis notebooks first to generate the required data files.")
        return
    st.sidebar.title("Filters")
    sentiments = ['All'] + df['title_sentiment'].unique().tolist()
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    all_sources = df['source_name'].value_counts().index.tolist()
    selected_sources = st.sidebar.multiselect("Sources", options=all_sources, default=[])
    clusters = ['All'] + sorted(df['kmeans_cluster'].unique().tolist())
    selected_cluster = st.sidebar.selectbox("Topic Cluster", clusters)
    df_filtered = df.copy()
    if selected_sentiment != 'All':
        df_filtered = df_filtered[df_filtered['title_sentiment'] == selected_sentiment]
    if selected_sources:
        df_filtered = df_filtered[df_filtered['source_name'].isin(selected_sources)]
    if selected_cluster != 'All':
        df_filtered = df_filtered[df_filtered['kmeans_cluster'] == selected_cluster]
    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Articles", f"{len(df_filtered):,}")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Live Inference", "Overview", "Semantic Map", "Temporal Analysis",
        "Source Comparison", "Article Explorer", "Model Performance", "BERTopic Analysis"
    ])
    with tab1:
        render_live_inference(df)
    with tab2:
        render_overview(df_filtered)
    with tab3:
        render_semantic_map(df_filtered, embeddings_2d, df)
    with tab4:
        render_temporal(df_temporal)
    with tab5:
        render_source_comparison(df_filtered)
    with tab6:
        render_article_explorer(df_filtered)
    with tab7:
        render_model_performance()
    with tab8:
        render_bertopic_analysis()


@st.cache_data
def load_raw_data_sample():
    base_path = Path(__file__).parent
    raw_path = base_path / 'raw-data.csv'
    if raw_path.exists():
        df = pd.read_csv(raw_path, usecols=['title'], nrows=50000)
        df = df.dropna(subset=['title'])
        df = df[df['title'].str.len() > 20]
        df = df[df['title'].str.len() < 200]
        return df['title'].tolist()
    return []


def render_live_inference(df):
    st.header("Live News Analysis")
    st.markdown("Enter any news headline to analyze its sentiment, topic, and find similar articles from the dataset.")
    models = load_inference_models()
    examples = {
        "-- Select an example --": "",
        "Russia launches massive drone attack on Ukraine": "Russia launches massive drone attack on Ukraine's capital",
        "Israel-Hamas conflict escalates in Gaza": "Israel and Hamas fighting intensifies as ceasefire talks fail",
        "China warns Taiwan over military exercises": "China issues strong warning to Taiwan amid military drills",
        "UN Security Council meets on Syria crisis": "UN Security Council holds emergency meeting on Syria humanitarian crisis",
        "Apple announces record-breaking iPhone sales": "Apple announces record-breaking iPhone sales in Q4",
        "Tesla stock surges after earnings beat": "Tesla shares jump 15% after quarterly earnings exceed expectations",
        "Amazon reports strong holiday shopping season": "Amazon posts record revenue during Black Friday and Cyber Monday",
        "Microsoft acquires gaming company for $10 billion": "Microsoft announces $10 billion acquisition of major gaming studio",
        "Stock market closes higher amid economic data": "Stock market closes higher amid positive economic data",
        "Federal Reserve holds interest rates steady": "Federal Reserve decides to hold interest rates unchanged",
        "Oil prices stable as OPEC meets": "Oil prices remain stable ahead of crucial OPEC meeting",
        "NYSE trading volume hits monthly average": "New York Stock Exchange reports average trading volumes",
        "Scientists make breakthrough in cancer treatment": "Scientists discover promising new approach to cure cancer",
        "Climate summit reaches historic agreement": "World leaders reach landmark climate deal at COP summit",
        "NASA discovers high potential for life on Mars": "NASA announces evidence suggesting conditions for life on Mars",
        "Police investigate shooting in downtown area": "Police launch investigation after fatal shooting in city center",
        "Major data breach affects millions of users": "Hackers expose personal data of 50 million users in security breach",
        "World Cup final breaks viewership records": "FIFA World Cup final becomes most watched sporting event in history",
        "Modi announces new economic reforms": "PM Modi unveils major economic reform package for India"
    }
    col_select, col_random = st.columns([3, 1])
    with col_select:
        selected_example = st.selectbox("Quick examples (or type your own below):", options=list(examples.keys()), index=0)
    with col_random:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Random from raw-data"):
            raw_titles = load_raw_data_sample()
            if raw_titles:
                import random
                random_title = random.choice(raw_titles)
                st.session_state['random_headline'] = random_title
    if 'random_headline' in st.session_state and st.session_state['random_headline']:
        default_text = st.session_state['random_headline']
        st.session_state['random_headline'] = ""
    else:
        default_text = examples.get(selected_example, "")
    headline = st.text_input("Enter a news headline:", value=default_text, placeholder="Type or paste a news headline here...")
    if headline:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Sentiment Analysis")
            if 'tfidf' in models and 'logreg' in models:
                processed = headline.lower().strip()
                X = models['tfidf'].transform([processed])
                prediction = models['logreg'].predict(X)[0]
                probabilities = models['logreg'].predict_proba(X)[0]
                sentiment_label = models['reverse_mapping'].get(prediction, str(prediction))
                confidence = probabilities[prediction] * 100
                sentiment_colors = {'Negative': '#e74c3c', 'Neutral': '#3498db', 'Positive': '#2ecc71'}
                sentiment_emojis = {'Negative': '', 'Neutral': '', 'Positive': ''}
                color = sentiment_colors.get(sentiment_label, '#gray')
                emoji = sentiment_emojis.get(sentiment_label, '')
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                    <h2 style="color: {color}; margin: 0;">{emoji} {sentiment_label}</h2>
                    <p style="font-size: 1.2rem; margin: 10px 0 0 0;">Confidence: <strong>{confidence:.1f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Probability Distribution:**")
                prob_df = pd.DataFrame({'Sentiment': ['Negative', 'Neutral', 'Positive'], 'Probability': probabilities * 100})
                fig = px.bar(prob_df, x='Sentiment', y='Probability', color='Sentiment', color_discrete_map=sentiment_colors, text=[f"{p:.1f}%" for p in prob_df['Probability']])
                fig.update_traces(textposition='outside')
                fig.update_layout(height=250, showlegend=False, yaxis_range=[0, 110], yaxis_title="Probability (%)")
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Sentiment model not loaded. Please ensure model files exist.")
        with col2:
            st.subheader("Topic Detection")
            try:
                topics_df, labels, bertopic_df = load_bertopic_data()
                use_semantic = st.checkbox("Use semantic matching (based on BERTopic)", value=True)
                if use_semantic and models.get('sentence_transformer') and bertopic_df is not None and labels:
                    headline_embedding = models['sentence_transformer'].encode([headline])[0]
                    valid_topics = bertopic_df[bertopic_df['bertopic_topic'] >= 0]
                    embeddings_path = Path(__file__).parent / 'outputs' / 'embeddings_minilm.npy'
                    if embeddings_path.exists() and 'embeddings' in models:
                        all_embeddings = models['embeddings']
                        topic_centroids = {}
                        for topic_id in valid_topics['bertopic_topic'].unique():
                            topic_indices = valid_topics[valid_topics['bertopic_topic'] == topic_id].index.tolist()
                            topic_indices = [i for i in topic_indices if i < len(all_embeddings)]
                            if topic_indices:
                                centroid = np.mean(all_embeddings[topic_indices], axis=0)
                                topic_centroids[topic_id] = centroid
                        best_topic_id = -1
                        best_similarity = -1
                        for topic_id, centroid in topic_centroids.items():
                            similarity = np.dot(headline_embedding, centroid) / (np.linalg.norm(headline_embedding) * np.linalg.norm(centroid))
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_topic_id = topic_id
                        if best_topic_id >= 0:
                            topic_label = labels.get(str(best_topic_id), f"Topic {best_topic_id}")
                            topic_row = topics_df[topics_df['Topic'] == best_topic_id]
                            if not topic_row.empty and 'Representation' in topic_row.columns:
                                rep = topic_row.iloc[0]['Representation']
                                if isinstance(rep, str) and rep.startswith('['):
                                    topic_keywords = ", ".join(eval(rep)[:5])
                                else:
                                    topic_keywords = str(rep)[:50]
                            else:
                                topic_keywords = "N/A"
                            best_color = "#9b59b6"
                            confidence = best_similarity * 100
                            st.markdown(f"""
                            <div style="background-color: {best_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {best_color};">
                                <h3 style="color: {best_color}; margin: 0;">{topic_label}</h3>
                                <p style="margin: 10px 0 0 0;"><strong>Top keywords:</strong> {topic_keywords}</p>
                                <p style="margin: 5px 0 0 0;"><strong>Similarity:</strong> {confidence:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.caption("Using semantic similarity to BERTopic clusters")
                        else:
                            raise ValueError("No matching topic found")
                    else:
                        raise ValueError("Embeddings not available")
                else:
                    headline_lower = headline.lower()
                    topic_keywords_map = {
                        "Gaza / Israel / Israeli": {"keywords": ["gaza", "israel", "hamas", "palestinian", "israeli", "idf", "netanyahu", "tel aviv", "west bank", "hostage"], "color": "#e74c3c"},
                        "Ukraine / Russian / Russia": {"keywords": ["ukraine", "russia", "russian", "putin", "kyiv", "moscow", "kremlin", "zelensky", "drone", "missile"], "color": "#e74c3c"},
                        "Market / Business / Growth": {"keywords": ["market", "billion", "usd", "growth", "revenue", "profit", "economy", "gdp", "trade", "investment"], "color": "#3498db"},
                        "Stocks / Finance / Trading": {"keywords": ["shares", "stock", "holdings", "nasdaq", "nyse", "trading", "investor", "dividend", "earnings", "quarterly"], "color": "#3498db"},
                        "Climate / Environment": {"keywords": ["climate", "environment", "emissions", "warming", "carbon", "renewable", "energy", "solar", "cop", "sustainability"], "color": "#27ae60"},
                        "Crime / Police / Violence": {"keywords": ["murder", "police", "killed", "death", "crime", "shooting", "arrested", "prison", "suspect", "investigation"], "color": "#e74c3c"},
                        "Black Friday / Shopping / Deals": {"keywords": ["black friday", "deals", "sale", "discount", "shopping", "cyber monday", "holiday", "gift", "amazon", "retail"], "color": "#2ecc71"},
                        "China / Asia Pacific": {"keywords": ["china", "chinese", "beijing", "taiwan", "xi jinping", "hong kong", "asia", "pacific", "south china sea"], "color": "#9b59b6"},
                        "Sports / World Cup": {"keywords": ["world cup", "football", "soccer", "fifa", "team", "match", "championship", "cricket", "sports", "player"], "color": "#f39c12"},
                        "India / Modi / Politics": {"keywords": ["modi", "india", "congress", "bjp", "delhi", "mumbai", "indian", "rupee", "pradesh"], "color": "#e67e22"},
                        "US Politics / Trump / Biden": {"keywords": ["trump", "republicans", "democrats", "biden", "congress", "senate", "house", "election", "white house", "gop"], "color": "#3498db"},
                        "Technology / AI / Innovation": {"keywords": ["ai", "artificial intelligence", "tech", "apple", "google", "microsoft", "tesla", "iphone", "software", "app"], "color": "#1abc9c"},
                        "Health / Science / Research": {"keywords": ["cancer", "health", "medical", "research", "scientist", "study", "treatment", "vaccine", "disease", "breakthrough"], "color": "#2ecc71"},
                        "Entertainment / Movies / TV": {"keywords": ["movie", "film", "netflix", "series", "actor", "actress", "hollywood", "box office", "streaming", "show"], "color": "#9b59b6"}
                    }
                    best_topic = None
                    best_score = 0
                    best_color = "#9b59b6"
                    matched_words = []
                    for topic_name, topic_data in topic_keywords_map.items():
                        keywords = topic_data["keywords"]
                        matches = [kw for kw in keywords if kw in headline_lower]
                        score = len(matches)
                        if score > best_score:
                            best_score = score
                            best_topic = topic_name
                            best_color = topic_data["color"]
                            matched_words = matches
                    if best_topic and best_score > 0:
                        topic_label = best_topic
                        topic_keywords = ", ".join(matched_words)
                    else:
                        topic_label = "General News"
                        topic_keywords = "No specific topic detected"
                        best_color = "#95a5a6"
                    st.markdown(f"""
                    <div style="background-color: {best_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {best_color};">
                        <h3 style="color: {best_color}; margin: 0;">{topic_label}</h3>
                        <p style="margin: 10px 0 0 0;"><strong>Matched keywords:</strong> {topic_keywords}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Using keyword matching")
            except Exception:
                headline_lower = headline.lower()
                simple_topics = {
                    "Geopolitics / War": ["war", "attack", "military", "conflict", "israel", "gaza", "ukraine", "russia"],
                    "Business / Finance": ["market", "stock", "shares", "billion", "revenue", "nasdaq", "nyse"],
                    "Technology": ["apple", "google", "microsoft", "ai", "tech", "iphone", "software"],
                    "Climate": ["climate", "environment", "emissions", "carbon"],
                    "Crime": ["murder", "police", "shooting", "crime", "arrested"]
                }
                detected = "General News"
                for topic, keywords in simple_topics.items():
                    if any(kw in headline_lower for kw in keywords):
                        detected = topic
                        break
                st.markdown(f"""
                <div style="background-color: #95a5a620; padding: 20px; border-radius: 10px; border-left: 5px solid #95a5a6;">
                    <h3 style="color: #95a5a6; margin: 0;">{detected}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Using basic keyword matching")
        st.markdown("---")
        st.subheader("Similar Articles from Dataset")
        if models.get('sentence_transformer') and 'embeddings' in models:
            try:
                headline_embedding = models['sentence_transformer'].encode([headline])
                similarities = cosine_similarity(headline_embedding, models['embeddings'])[0]
                top_indices = np.argsort(similarities)[-6:][::-1]
                similar_articles = []
                for idx in top_indices:
                    if idx < len(df):
                        sim_score = similarities[idx]
                        article = df.iloc[idx]
                        if sim_score < 0.99:
                            similar_articles.append({
                                'Title': article['title'],
                                'Source': article.get('source_name', 'Unknown'),
                                'Sentiment': article.get('title_sentiment', 'Unknown'),
                                'Similarity': f"{sim_score*100:.1f}%"
                            })
                        if len(similar_articles) >= 5:
                            break
                if similar_articles:
                    similar_df = pd.DataFrame(similar_articles)
                    def highlight_sentiment(val):
                        if val == 'Negative':
                            return 'background-color: #ffcccc'
                        elif val == 'Positive':
                            return 'background-color: #ccffcc'
                        return ''
                    styled_df = similar_df.style.map(highlight_sentiment, subset=['Sentiment'])
                    st.dataframe(styled_df, width="stretch", hide_index=True)
                else:
                    st.info("No similar articles found.")
            except Exception as e:
                st.warning(f"Similarity search error: {str(e)}")
        else:
            try:
                headline_words = set(headline.lower().split())
                matches = []
                for idx, row in df.head(5000).iterrows():
                    title = str(row.get('title', '')).lower()
                    title_words = set(title.split())
                    common = len(headline_words & title_words)
                    if common >= 2:
                        matches.append({
                            'idx': idx,
                            'score': common,
                            'Title': row['title'],
                            'Source': row.get('source_name', 'Unknown'),
                            'Sentiment': row.get('title_sentiment', 'Unknown')
                        })
                matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:5]
                if matches:
                    similar_df = pd.DataFrame(matches)[['Title', 'Source', 'Sentiment']]
                    def highlight_sentiment(val):
                        if val == 'Negative':
                            return 'background-color: #ffcccc'
                        elif val == 'Positive':
                            return 'background-color: #ccffcc'
                        return ''
                    styled_df = similar_df.style.map(highlight_sentiment, subset=['Sentiment'])
                    st.dataframe(styled_df, width="stretch", hide_index=True)
                    st.caption("*Using keyword matching (semantic search unavailable)*")
                else:
                    st.info("No similar articles found by keyword matching.")
            except Exception as e:
                st.warning(f"Keyword search error: {str(e)}")


def render_overview(df):
    st.header("Dataset Overview")
    if len(df) == 0:
        st.warning("No articles match the current filters. Please adjust your selection.")
        return
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", f"{len(df):,}")
    with col2:
        neg_pct = (df['title_sentiment'] == 'Negative').sum() / len(df) * 100
        st.metric("Negative %", f"{neg_pct:.1f}%")
    with col3:
        n_sources = df['source_name'].nunique()
        st.metric("Sources", n_sources)
    with col4:
        n_clusters = df['kmeans_cluster'].nunique()
        st.metric("Topic Clusters", n_clusters)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sentiment_counts = df['title_sentiment'].value_counts()
        colors = {'Negative': '#e74c3c', 'Neutral': '#3498db', 'Positive': '#2ecc71'}
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution", color=sentiment_counts.index, color_discrete_map=colors)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, width="stretch")
    with col2:
        cluster_counts = df['kmeans_cluster'].value_counts().sort_index()
        cluster_names = {0: 'Business/Markets', 1: 'Stocks/Finance', 2: 'Geopolitics/War'}
        fig = px.bar(x=[cluster_names.get(c, f'Cluster {c}') for c in cluster_counts.index], y=cluster_counts.values, title="Topic Cluster Distribution", labels={'x': 'Cluster', 'y': 'Articles'}, color=cluster_counts.index.astype(str), color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, width="stretch")
    st.subheader("Top Sources")
    top_sources = df['source_name'].value_counts().head(10)
    source_sentiment = df.groupby('source_name').agg({'title_sentiment': lambda x: (x == 'Negative').sum() / len(x) * 100}).rename(columns={'title_sentiment': 'negative_pct'})
    source_df = pd.DataFrame({'Source': top_sources.index, 'Articles': top_sources.values, 'Negative %': [source_sentiment.loc[s, 'negative_pct'] for s in top_sources.index]})
    fig = px.bar(source_df, x='Articles', y='Source', orientation='h', color='Negative %', color_continuous_scale='RdYlGn_r', title="Top 10 Sources by Volume (colored by negative sentiment %)")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, width="stretch")


def render_semantic_map(df_filtered, embeddings_2d, df_full):
    st.header("Semantic Map")
    if embeddings_2d is None:
        st.warning("UMAP embeddings not found. Please run the clustering notebook first.")
        return
    if len(df_filtered) == 0:
        st.warning("No articles match the current filters. Please adjust your selection.")
        return
    filtered_indices = df_filtered.index.tolist()
    max_points = 5000
    if len(filtered_indices) > max_points:
        filtered_indices = np.random.choice(filtered_indices, max_points, replace=False)
        st.info(f"Showing {max_points:,} randomly sampled points for performance.")
    color_by = st.radio("Color by:", ["Sentiment", "Cluster"], horizontal=True)
    plot_df = df_full.iloc[filtered_indices].copy()
    plot_df['x'] = embeddings_2d[filtered_indices, 0]
    plot_df['y'] = embeddings_2d[filtered_indices, 1]
    if color_by == "Sentiment":
        color_col = 'title_sentiment'
        color_map = {'Negative': '#e74c3c', 'Neutral': '#3498db', 'Positive': '#2ecc71'}
    else:
        plot_df['cluster_name'] = plot_df['kmeans_cluster'].map({0: 'Business/Markets', 1: 'Stocks/Finance', 2: 'Geopolitics/War'})
        color_col = 'cluster_name'
        color_map = None
    fig = px.scatter(plot_df, x='x', y='y', color=color_col, color_discrete_map=color_map, hover_data=['title', 'source_name'], title="UMAP Projection of News Articles", opacity=0.6)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(xaxis_title="UMAP 1", yaxis_title="UMAP 2", height=600)
    st.plotly_chart(fig, width="stretch")


def render_temporal(df_temporal):
    st.header("Source Sentiment Over Time")
    if df_temporal is None:
        st.warning("Temporal data not found. Please run the temporal analysis notebook first.")
        return
    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df_temporal['sentiment_score'] = df_temporal['title_sentiment'].map(sentiment_map)
    top_sources = df_temporal['source_name'].value_counts().head(6).index.tolist()
    source_daily = df_temporal[df_temporal['source_name'].isin(top_sources)].groupby(['date', 'source_name'])['sentiment_score'].mean().reset_index()
    fig = px.line(source_daily, x='date', y='sentiment_score', color='source_name', title="Sentiment by Source Over Time")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")


def render_source_comparison(df):
    st.header("Source Comparison")
    if len(df) == 0:
        st.warning("No articles match the current filters. Please adjust your selection.")
        return
    source_stats = df.groupby('source_name').agg({
        'title': 'count',
        'title_sentiment': [lambda x: (x == 'Negative').sum() / len(x) * 100, lambda x: (x == 'Positive').sum() / len(x) * 100],
        'kmeans_cluster': lambda x: x.mode()[0] if len(x) > 0 else None
    }).reset_index()
    source_stats.columns = ['source', 'articles', 'negative_pct', 'positive_pct', 'dominant_cluster']
    source_stats = source_stats.sort_values('articles', ascending=False).head(15)
    cluster_names = {0: 'Business', 1: 'Finance', 2: 'Geopolitics'}
    source_stats['cluster_name'] = source_stats['dominant_cluster'].map(cluster_names)
    fig = px.scatter(source_stats, x='negative_pct', y='positive_pct', size='articles', color='cluster_name', hover_name='source', title="Source Positioning: Negative vs Positive Sentiment", labels={'negative_pct': 'Negative %', 'positive_pct': 'Positive %', 'cluster_name': 'Dominant Topic'})
    fig.add_vline(x=source_stats['negative_pct'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=source_stats['positive_pct'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Source Details")
    display_df = source_stats[['source', 'articles', 'negative_pct', 'positive_pct', 'cluster_name']].copy()
    display_df.columns = ['Source', 'Articles', 'Negative %', 'Positive %', 'Dominant Topic']
    display_df['Negative %'] = display_df['Negative %'].round(1)
    display_df['Positive %'] = display_df['Positive %'].round(1)
    st.dataframe(display_df, width="stretch", hide_index=True)


def render_article_explorer(df):
    st.header("Article Explorer")
    if len(df) == 0:
        st.warning("No articles match the current filters. Please adjust your selection.")
        return
    search_query = st.text_input("Search in titles", "")
    if search_query:
        df_search = df[df['title'].str.contains(search_query, case=False, na=False)]
    else:
        df_search = df
    st.write(f"Found {len(df_search):,} articles")
    n_show = st.slider("Articles to show", 10, 100, 25)
    display_cols = ['title', 'source_name', 'title_sentiment', 'kmeans_cluster']
    available_cols = [c for c in display_cols if c in df_search.columns]
    display_df = df_search[available_cols].head(n_show).copy()
    if 'kmeans_cluster' in display_df.columns:
        cluster_names = {0: 'Business/Markets', 1: 'Stocks/Finance', 2: 'Geopolitics/War'}
        display_df['Topic'] = display_df['kmeans_cluster'].map(cluster_names)
        display_df = display_df.drop(columns=['kmeans_cluster'])
    display_df.columns = ['Title', 'Source', 'Sentiment', 'Topic']
    def highlight_sentiment(val):
        if val == 'Negative':
            return 'background-color: #ffcccc'
        elif val == 'Positive':
            return 'background-color: #ccffcc'
        return ''
    styled_df = display_df.style.map(highlight_sentiment, subset=['Sentiment'])
    st.dataframe(styled_df, width="stretch", hide_index=True)


def render_model_performance():
    st.header("Model Performance Comparison")
    results = load_model_results()
    if not results:
        st.warning("No model results found. Please run the model evaluation notebooks first.")
        return
    st.subheader("Model Evaluation Summary")
    models_data = []
    if 'temporal' in results:
        temporal = results['temporal']
        for model_name, metrics in temporal['models'].items():
            display_name = {'tfidf_logreg': 'TF-IDF + LogReg', 'embeddings_logreg': 'Embeddings + LogReg', 'ensemble_concat': 'Ensemble (Concat)', 'ensemble_voting': 'Ensemble (Voting)'}.get(model_name, model_name)
            models_data.append({'Model': display_name, 'Accuracy': metrics['accuracy'], 'F1-macro': metrics['f1_macro'], 'Split Type': 'Temporal (Oct→Nov)', 'is_best': model_name == 'ensemble_voting'})
    if 'bert' in results:
        bert = results['bert']
        models_data.append({'Model': 'BERT Fine-tuned', 'Accuracy': bert['test_accuracy'], 'F1-macro': bert['test_f1_macro'], 'Split Type': 'Random (70/15/15)', 'is_best': False})
    if models_data:
        models_df = pd.DataFrame(models_data)
        fig = go.Figure()
        colors = ['#3498db' if not row['is_best'] else '#e74c3c' for _, row in models_df.iterrows()]
        fig.add_trace(go.Bar(name='Accuracy', x=models_df['Model'], y=models_df['Accuracy'], marker_color='#3498db', text=[f"{v:.1%}" for v in models_df['Accuracy']], textposition='outside'))
        fig.add_trace(go.Bar(name='F1-macro', x=models_df['Model'], y=models_df['F1-macro'], marker_color='#2ecc71', text=[f"{v:.1%}" for v in models_df['F1-macro']], textposition='outside'))
        fig.update_layout(title="Model Performance Comparison", barmode='group', yaxis_title="Score", yaxis_range=[0, 1.1], height=450)
        st.plotly_chart(fig, width="stretch")
        display_models = models_df[['Model', 'Accuracy', 'F1-macro', 'Split Type']].copy()
        display_models['Accuracy'] = display_models['Accuracy'].apply(lambda x: f"{x:.1%}")
        display_models['F1-macro'] = display_models['F1-macro'].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_models, width="stretch", hide_index=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Temporal Split Results")
        if 'temporal' in results:
            temporal = results['temporal']
            st.markdown(f"""
            - **Train Period**: {temporal['train_period']}
            - **Test Period**: {temporal['test_period']}
            - **Train Samples**: {temporal['train_samples']:,}
            - **Test Samples**: {temporal['test_samples']:,}
            - **Best Model**: {temporal['best_model']}
            - **Best F1**: {temporal['best_f1_macro']:.1%}
            """)
            st.info("Temporal split shows more realistic performance (~5-7% lower than random split) as it tests generalization to future data.")
        else:
            st.warning("Temporal split results not available.")
    with col2:
        st.subheader("BERT Fine-tuning Results")
        if 'bert' in results:
            bert = results['bert']
            st.markdown(f"""
            - **Base Model**: `{bert['model']}`
            - **Epochs**: {bert['epochs']}
            - **Batch Size**: {bert['batch_size']}
            - **Learning Rate**: {bert['learning_rate']}
            - **Test Accuracy**: {bert['test_accuracy']:.1%}
            - **Test F1-macro**: {bert['test_f1_macro']:.1%}
            """)
            st.warning("High BERT performance (~94%) is because the same model did the original labeling. This represents upper bound, not realistic production performance.")
        else:
            st.warning("BERT results not available.")


def render_bertopic_analysis():
    st.header("BERTopic Topic Analysis")
    topics_df, labels, bertopic_df = load_bertopic_data()
    if topics_df is None:
        st.warning("BERTopic results not found. Please run notebook 08_bertopic_analysis.ipynb first.")
        return
    col1, col2, col3 = st.columns(3)
    n_topics = len(topics_df[topics_df['Topic'] >= 0])
    outliers = topics_df[topics_df['Topic'] == -1]['Count'].values[0] if -1 in topics_df['Topic'].values else 0
    total_docs = topics_df['Count'].sum()
    with col1:
        st.metric("Topics Discovered", n_topics)
    with col2:
        st.metric("Outlier Documents", f"{outliers:,}")
    with col3:
        st.metric("Outlier Rate", f"{outliers/total_docs*100:.1f}%")
    st.markdown("---")
    st.subheader("Topic Distribution")
    topics_no_outliers = topics_df[topics_df['Topic'] >= 0].copy()
    topics_no_outliers = topics_no_outliers.sort_values('Count', ascending=False).head(15)
    if labels:
        topics_no_outliers['Label'] = topics_no_outliers['Topic'].astype(str).map(labels).fillna(topics_no_outliers['Name'])
    else:
        topics_no_outliers['Label'] = topics_no_outliers['Name']
    fig = px.bar(topics_no_outliers, x='Count', y='Label', orientation='h', title="Top 15 Topics by Document Count", color='Count', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Topic Sentiment Analysis")
    if bertopic_df is not None and 'bertopic_topic' in bertopic_df.columns:
        sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
        bertopic_df['sentiment_score'] = bertopic_df['title_sentiment'].map(sentiment_map)
        topic_sentiment = bertopic_df[bertopic_df['bertopic_topic'] >= 0].groupby('bertopic_topic').agg({'sentiment_score': 'mean', 'title': 'count'}).reset_index()
        topic_sentiment.columns = ['Topic', 'Mean Sentiment', 'Count']
        topic_sentiment = topic_sentiment.sort_values('Mean Sentiment')
        if labels:
            topic_sentiment['Label'] = topic_sentiment['Topic'].astype(str).map(labels).fillna(f"Topic {topic_sentiment['Topic']}")
        else:
            topic_sentiment['Label'] = topic_sentiment['Topic'].apply(lambda x: f"Topic {x}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most Negative Topics**")
            neg_topics = topic_sentiment.head(5)
            for _, row in neg_topics.iterrows():
                st.markdown(f"- **{row['Label']}**: {row['Mean Sentiment']:.2f} ({row['Count']:,} docs)")
        with col2:
            st.markdown("**Most Positive Topics**")
            pos_topics = topic_sentiment.tail(5).iloc[::-1]
            for _, row in pos_topics.iterrows():
                st.markdown(f"- **{row['Label']}**: {row['Mean Sentiment']:+.2f} ({row['Count']:,} docs)")
        fig = px.scatter(topic_sentiment, x='Mean Sentiment', y='Count', text='Label', title="Topic Map: Sentiment vs Volume", color='Mean Sentiment', color_continuous_scale='RdYlGn', size='Count', size_max=50)
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
    st.subheader("All Topics")
    topics_display = topics_df[topics_df['Topic'] >= 0][['Topic', 'Count', 'Name', 'Representation']].copy()
    if labels:
        topics_display['Label'] = topics_display['Topic'].astype(str).map(labels)
    topics_display = topics_display.sort_values('Count', ascending=False)
    topics_display['Top Keywords'] = topics_display['Representation'].apply(lambda x: ', '.join(eval(x)[:5]) if isinstance(x, str) and x.startswith('[') else str(x)[:50])
    display_cols = ['Topic', 'Label', 'Count', 'Top Keywords'] if labels else ['Topic', 'Count', 'Top Keywords']
    st.dataframe(topics_display[display_cols], width="stretch", hide_index=True)
    st.markdown("---")
    st.subheader("Interactive Visualizations")
    base_path = Path(__file__).parent
    viz_files = {'Topic Keywords': 'bertopic_barchart.html', 'Topic Hierarchy': 'bertopic_hierarchy.html', 'Topic Similarity': 'bertopic_heatmap.html'}
    selected_viz = st.selectbox("Select Visualization", list(viz_files.keys()))
    viz_path = base_path / 'outputs' / viz_files[selected_viz]
    if viz_path.exists():
        with open(viz_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.warning(f"Visualization file not found: {viz_files[selected_viz]}")


if __name__ == "__main__":
    main()
