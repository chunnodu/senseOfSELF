import streamlit as st
import pandas as pd
import json
from pathlib import Path
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Load and process Twitter data
@st.cache_data
def load_twitter_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove the JavaScript assignment at the beginning
    json_str = content.split('=', 1)[1].strip()
    
    # Remove trailing semicolon if present
    if json_str.endswith(';'):
        json_str = json_str[:-1]
    
    data = json.loads(json_str)
    
    # Extract tweets from the new structure
    tweets = []
    for tweet in data:
        tweet_data = tweet['tweet']
        tweets.append({
            'id': tweet_data['id'],
            'created_at': tweet_data['created_at'],
            'full_text': tweet_data['full_text'] if 'full_text' in tweet_data else tweet_data['text'],
        })
    
    df = pd.DataFrame(tweets)
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df

# Analyze tweet content for Sense of Self
def analyze_self_expression(tweets):
    docs = list(nlp.pipe(tweets, batch_size=100))
    
    entities = [ent.label_ for doc in docs for ent in doc.ents]
    lemmas = [token.lemma_ for doc in docs for token in doc 
              if not token.is_stop and not token.is_punct and token.is_alpha]
    
    return Counter(entities), Counter(lemmas)

# Generate word cloud
def generate_wordcloud(lemmas_count):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(lemmas_count)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Analyze data for a specific year
def analyze_year(df_year):
    tweet_counts = df_year.resample('M', on='created_at').size()
    entities_count, lemmas_count = analyze_self_expression(df_year['full_text'].tolist())
    top_entities = pd.DataFrame(entities_count.most_common(10), columns=['Subject', 'Frequency'])
    wordcloud_fig = generate_wordcloud(lemmas_count)
    
    return tweet_counts, top_entities, wordcloud_fig

# Main app
def main():
    st.title("Sense of Self: Yearly Twitter Archive Analysis")

    try:
        df = load_twitter_data("tweets.js")
        
        st.header("Your Twitter Journey")
        st.write(f"Total Tweets: {len(df)}")
        st.write(f"Date Range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")
        
        years = df['created_at'].dt.year.unique()
        years.sort()
        
        for year in years:
            st.header(f"Your {year} in Tweets")
            df_year = df[df['created_at'].dt.year == year]
            
            tweet_counts, top_entities, wordcloud_fig = analyze_year(df_year)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Your Tweeting Rhythm")
                st.line_chart(tweet_counts)
                
                st.subheader("Top Subjects in Your Tweets")
                st.bar_chart(top_entities.set_index('Subject'))
            
            with col2:
                st.subheader("Your Twitter Vocabulary")
                st.pyplot(wordcloud_fig)
            
            st.write("---")
    
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")

if __name__ == "__main__":
    main()