import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from collections import Counter
import re

def clean_text(text):
    """Simple text cleaning function"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def calculate_tf(doc):
    """Calculate term frequency for a document"""
    words = doc.split()
    word_count = len(words)
    tf_dict = {}
    for word in words:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    
    # Normalize by total word count
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / word_count
    
    return tf_dict

def calculate_idf(documents):
    """Calculate inverse document frequency"""
    N = len(documents)
    idf_dict = {}
    all_words = set(word for doc in documents for word in doc.split())
    
    for word in all_words:
        containing_docs = sum(1 for doc in documents if word in doc.split())
        idf_dict[word] = math.log(N / containing_docs)
    
    return idf_dict

def calculate_tfidf_manual(documents):
    """Manual TF-IDF calculation for educational purposes"""
    # Calculate TF for each document
    tf_docs = []
    for doc in documents:
        tf_docs.append(calculate_tf(doc))
    
    # Calculate IDF
    idf_dict = calculate_idf(documents)
    
    # Calculate TF-IDF
    tfidf_docs = []
    for tf_doc in tf_docs:
        tfidf_doc = {}
        for word, tf_val in tf_doc.items():
            tfidf_doc[word] = tf_val * idf_dict[word]
        tfidf_docs.append(tfidf_doc)
    
    return tf_docs, idf_dict, tfidf_docs

def show_tfidf_app():
    st.title("📊 TF-IDF: Term Frequency-Inverse Document Frequency")
    
    # Educational content
    st.markdown("""
    ## What is TF-IDF?
    
    **TF-IDF** is a numerical statistic that reflects how important a word is to a document in a collection of documents. It's widely used in text mining and information retrieval.
    
    ### The Formula:
    **TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)**
    
    Where:
    - **TF(t,d)** = Term Frequency of term t in document d
    - **IDF(t,D)** = Inverse Document Frequency of term t in document collection D
    
    ### Components:
    1. **Term Frequency (TF)**: How frequently a term appears in a document
       - TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)
    
    2. **Inverse Document Frequency (IDF)**: How rare or common a term is across all documents
       - IDF(t,D) = log(Total number of documents / Number of documents containing term t)
    
    ### Why TF-IDF?
    - **High TF**: Term appears frequently in the document (important to that document)
    - **High IDF**: Term is rare across the collection (distinctive)
    - **High TF-IDF**: Term is both frequent in the document AND rare in the collection (very important!)
    """)
    
    st.divider()
    
    # Input section
    st.header("🔧 Try It Yourself!")
    
    # Sample documents
    sample_docs = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "Cats and dogs are pets",
        "The park has many trees and flowers"
    ]
    
    input_method = st.radio("Choose input method:", ["Use sample documents", "Enter your own documents"])
    
    if input_method == "Use sample documents":
        documents = sample_docs
        st.write("**Sample Documents:**")
        for i, doc in enumerate(documents, 1):
            st.write(f"{i}. {doc}")
    else:
        st.write("Enter your documents (one per line):")
        user_input = st.text_area("Documents", height=150, 
                                 placeholder="Enter each document on a new line...\nExample:\nThe cat sat on the mat\nThe dog ran in the park")
        
        if user_input.strip():
            documents = [line.strip() for line in user_input.split('\n') if line.strip()]
        else:
            documents = sample_docs
            st.info("Using sample documents since no input provided.")
    
    if documents:
        # Clean documents
        cleaned_docs = [clean_text(doc) for doc in documents]
        
        st.divider()
        
        # Manual calculation
        st.header("📝 Step-by-Step Calculation")
        
        tf_docs, idf_dict, tfidf_docs = calculate_tfidf_manual(cleaned_docs)
        
        # Show TF calculation
        st.subheader("1. Term Frequency (TF) Calculation")
        st.write("TF = (Number of times term appears in document) / (Total terms in document)")
        
        tf_data = []
        for i, (doc, tf_doc) in enumerate(zip(documents, tf_docs)):
            for word, tf_val in tf_doc.items():
                tf_data.append({
                    'Document': f'Doc {i+1}',
                    'Original': doc,
                    'Term': word,
                    'TF': round(tf_val, 4)
                })
        
        tf_df = pd.DataFrame(tf_data)
        
        # Create TF pivot table
        tf_pivot = tf_df.pivot_table(index='Term', columns='Document', values='TF', fill_value=0)
        st.write("**Term Frequency Matrix:**")
        st.dataframe(tf_pivot.style.format("{:.4f}"))
        
        # Show IDF calculation
        st.subheader("2. Inverse Document Frequency (IDF) Calculation")
        st.write("IDF = log(Total documents / Documents containing term)")
        
        idf_data = []
        total_docs = len(documents)
        for word, idf_val in idf_dict.items():
            docs_containing = sum(1 for doc in cleaned_docs if word in doc.split())
            idf_data.append({
                'Term': word,
                'Documents Containing Term': docs_containing,
                'Total Documents': total_docs,
                'IDF': round(idf_val, 4)
            })
        
        idf_df = pd.DataFrame(idf_data).sort_values('IDF', ascending=False)
        st.dataframe(idf_df)
        
        # Show TF-IDF calculation
        st.subheader("3. TF-IDF Calculation")
        st.write("TF-IDF = TF × IDF")
        
        tfidf_data = []
        for i, (doc, tfidf_doc) in enumerate(zip(documents, tfidf_docs)):
            for word, tfidf_val in tfidf_doc.items():
                tf_val = tf_docs[i][word]
                idf_val = idf_dict[word]
                tfidf_data.append({
                    'Document': f'Doc {i+1}',
                    'Term': word,
                    'TF': round(tf_val, 4),
                    'IDF': round(idf_val, 4),
                    'TF-IDF': round(tfidf_val, 4)
                })
        
        tfidf_df = pd.DataFrame(tfidf_data)
        
        # Create TF-IDF pivot table
        tfidf_pivot = tfidf_df.pivot_table(index='Term', columns='Document', values='TF-IDF', fill_value=0)
        st.write("**TF-IDF Matrix:**")
        st.dataframe(tfidf_pivot.style.format("{:.4f}"))
        
        st.divider()
        
        # Visualizations
        st.header("📈 Visualizations")
        
        # TF-IDF Heatmap
        st.subheader("TF-IDF Heatmap")
        fig_heatmap = px.imshow(tfidf_pivot.values, 
                               x=tfidf_pivot.columns, 
                               y=tfidf_pivot.index,
                               color_continuous_scale='Viridis',
                               title="TF-IDF Scores Heatmap")
        fig_heatmap.update_layout(height=max(400, len(tfidf_pivot.index) * 30))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Top terms per document
        st.subheader("Top Terms per Document")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_doc = st.selectbox("Select document:", 
                                      [f"Doc {i+1}: {doc[:30]}..." for i, doc in enumerate(documents)])
            doc_idx = int(selected_doc.split()[1]) - 1
        
        with col2:
            top_n = st.slider("Number of top terms:", 3, 10, 5)
        
        # Get top terms for selected document
        doc_tfidf = tfidf_pivot.iloc[:, doc_idx].sort_values(ascending=False).head(top_n)
        
        fig_bar = px.bar(x=doc_tfidf.values, y=doc_tfidf.index, orientation='h',
                        title=f"Top {top_n} Terms in {selected_doc.split(':')[0]}",
                        labels={'x': 'TF-IDF Score', 'y': 'Terms'})
        fig_bar.update_layout(height=max(300, top_n * 40))
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Comparison with sklearn
        st.divider()
        st.header("🔬 Comparison with Scikit-learn")
        
        # Calculate using sklearn
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b[a-zA-Z]+\b')
        sklearn_tfidf = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        sklearn_df = pd.DataFrame(sklearn_tfidf.toarray(), 
                                 columns=feature_names,
                                 index=[f'Doc {i+1}' for i in range(len(documents))])
        
        st.write("**Scikit-learn TF-IDF Matrix:**")
        st.dataframe(sklearn_df.T.style.format("{:.4f}"))
        
        # Show differences
        st.subheader("Why might there be differences?")
        st.markdown("""
        - **Normalization**: Scikit-learn applies L2 normalization by default
        - **Smoothing**: Scikit-learn adds 1 to document frequencies to avoid division by zero
        - **Sublinear TF**: Option to use log(1 + tf) instead of raw tf
        - **Text preprocessing**: Different tokenization and cleaning methods
        """)
        
        st.divider()
        
        # Key insights
        st.header("🎯 Key Insights")
        
        # Find most distinctive terms
        max_tfidf_per_term = tfidf_pivot.max(axis=1).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Distinctive Terms")
            st.write("Terms with highest TF-IDF scores:")
            for term, score in max_tfidf_per_term.head(5).items():
                st.write(f"• **{term}**: {score:.4f}")
        
        with col2:
            st.subheader("Common Terms (Low IDF)")
            low_idf_terms = sorted(idf_dict.items(), key=lambda x: x[1])[:5]
            st.write("Terms appearing in many documents:")
            for term, idf_score in low_idf_terms:
                docs_with_term = sum(1 for doc in cleaned_docs if term in doc.split())
                st.write(f"• **{term}**: appears in {docs_with_term}/{len(documents)} docs")
        
        st.divider()
        
        # Educational summary
        st.header("📚 Summary")
        st.markdown("""
        ### What you learned:
        1. **TF-IDF combines two important concepts**: how often a term appears in a document (TF) and how rare it is across all documents (IDF)
        2. **High TF-IDF scores** indicate terms that are both frequent in a specific document and rare across the collection
        3. **Common words** (like "the", "and") get low TF-IDF scores because they appear in many documents
        4. **TF-IDF is the foundation** for many text analysis techniques and search engines
        
        ### Next steps in your learning journey:
        - **Bag of Words**: Learn about simpler text representation methods
        - **Word2Vec**: Discover how words can be represented as dense vectors
        - **Modern embeddings**: Explore BERT, GPT, and other transformer-based embeddings
        """)

if __name__ == "__main__":
    show_tfidf_app()

