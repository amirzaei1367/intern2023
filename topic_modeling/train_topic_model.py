import fire 
from bertopic import BERTopic 
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from transformers.pipelines import pipeline
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

def load_data(csv_data):
  df = pd.read_csv(csv_path)
  df = df.drop(columns=['Resume_html'])
  return df

def main(
  csv_path: str,
  output_dir: str,
  results: str,
  topic_results: str,
):
  df = load_data(csv_path)
  data= df['Resume_str'].tolist()

  embedding_model = pipeline(
    "feature-extraction",
    model="distilbert-base-cased",
    framework='pt',
    device=0,
  )
  kmeans_model = KMeans(
    n_clusters=20,
  )
  vectorizer_model = CountVectorizer(
    stop_words='english',
    ngram_range=(1,3),
    lowercase=True,
    max_df=.90,
    min_df=.10,
  )
  representation_model = MaximalMarginalRelevance()
  
  topic_model = BERTopic(
    language='english',
    nr_topics=20,
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    representation_model=represenatation_model,
    hdbscan_model=kmeans_model,
    verbose=True,
  )
  
  topic_model.fit(data)
  
  topic_model.save(
    output_dir,
    serialization='pytorch',
    save_embedding_model=True,
    save_ctfidf=True,
  )
  
  doc_df = topic_model.get_document_info(data)
  topic_df = topic_model.get_topic_info()
  
  print(doc_df)
  print(topic_df)
  
  id = df['ID']
  doc_df['ID'] = id
  
  doc_df.to_csv(results)
  topic_df.to_csv(topic_results)

if __name__ == "__main__":
  fire.Fire(main)
