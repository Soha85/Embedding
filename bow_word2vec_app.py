import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    """Simple text cleaning function"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def create_bow_manual(documents):
    """Manual Bag of Words implementation for educational purposes"""
    # Clean and tokenize documents
    cleaned_docs = [clean_text(doc) for doc in documents]
    tokenized_docs = [doc.split() for doc in cleaned_docs]
    
    # Create vocabulary
    vocabulary = set()
    for doc in tokenized_docs:
        vocabulary.update(doc)
    vocabulary = sorted(list(vocabulary))
    
    # Create BoW vectors
    bow_vectors = []
    for doc in tokenized_docs:
        vector = []
        for word in vocabulary:
            vector.append(doc.count(word))
        bow_vectors.append(vector)
    
    return vocabulary, bow_vectors, tokenized_docs

def show_bow_word2vec_app():
    st.title("🎒 Bag of Words & 🧠 Word2Vec")
    
    # Introduction
    st.markdown("""
    ## Understanding Text Representation Evolution
    
    This app demonstrates two fundamental approaches to representing text as numbers:
    1. **Bag of Words (BoW)**: Simple counting approach
    2. **Word2Vec**: Dense vector representations that capture semantic meaning
    """)
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Bag of Words", "🧠 Word2Vec", "🔄 Comparison"])
    
    # Input section (shared across tabs)
    st.sidebar.header("📝 Input Documents")
    
    sample_docs = [
        "The king is a strong ruler",
        "The queen is a wise leader", 
        "A man walks in the park",
        "A woman runs in the garden",
        "The cat sits on the mat",
        "The dog plays in the yard",
        "Programming is fun and creative",
        "Coding requires logic and patience"
    ]
    
    input_method = st.sidebar.radio("Choose input method:", ["Use sample documents", "Enter your own documents"])
    
    if input_method == "Use sample documents":
        documents = sample_docs
        st.sidebar.write("**Sample Documents:**")
        for i, doc in enumerate(documents, 1):
            st.sidebar.write(f"{i}. {doc}")
    else:
        st.sidebar.write("Enter your documents:")
        user_input = st.sidebar.text_area("Documents", height=200, 
                                         placeholder="Enter each document on a new line...")
        
        if user_input.strip():
            documents = [line.strip() for line in user_input.split('\n') if line.strip()]
        else:
            documents = sample_docs
            st.sidebar.info("Using sample documents.")
    
    # Tab 1: Bag of Words
    with tab1:
        st.header("🎒 Bag of Words (BoW)")
        
        st.markdown("""
        ## What is Bag of Words?
        
        **Bag of Words** is the simplest way to convert text into numbers. It treats each document as a "bag" containing words, ignoring:
        - Word order
        - Grammar
        - Context
        
        ### How it works:
        1. **Create vocabulary**: Collect all unique words from all documents
        2. **Count occurrences**: For each document, count how many times each word appears
        3. **Create vectors**: Each document becomes a vector where each position represents a word count
        
        ### Example:
        - Document 1: "The cat sat"
        - Document 2: "The dog ran"
        - Vocabulary: ["cat", "dog", "ran", "sat", "the"]
        - Doc 1 vector: [1, 0, 0, 1, 1] (cat=1, dog=0, ran=0, sat=1, the=1)
        - Doc 2 vector: [0, 1, 1, 0, 1] (cat=0, dog=1, ran=1, sat=0, the=1)
        """)
        
        if documents:
            st.divider()
            st.subheader("📊 Step-by-Step BoW Creation")
            
            # Manual BoW calculation
            vocabulary, bow_vectors, tokenized_docs = create_bow_manual(documents)
            
            # Show vocabulary
            st.write("**1. Vocabulary Creation**")
            st.write(f"Total unique words: {len(vocabulary)}")
            
            vocab_cols = st.columns(3)
            for i, word in enumerate(vocabulary):
                vocab_cols[i % 3].write(f"• {word}")
            
            # Show tokenized documents
            st.write("**2. Document Tokenization**")
            for i, (original, tokens) in enumerate(zip(documents, tokenized_docs)):
                with st.expander(f"Document {i+1}: {original[:50]}..."):
                    st.write(f"**Original**: {original}")
                    st.write(f"**Tokens**: {tokens}")
                    st.write(f"**Token count**: {len(tokens)}")
            
            # Show BoW matrix
            st.write("**3. Bag of Words Matrix**")
            bow_df = pd.DataFrame(bow_vectors, 
                                 columns=vocabulary,
                                 index=[f'Doc {i+1}' for i in range(len(documents))])
            
            st.dataframe(bow_df)
            
            # BoW visualization
            st.subheader("📈 BoW Visualization")
            
            # Heatmap
            fig_bow_heatmap = px.imshow(bow_df.values, 
                                       x=bow_df.columns, 
                                       y=bow_df.index,
                                       color_continuous_scale='Blues',
                                       title="Bag of Words Matrix Heatmap")
            fig_bow_heatmap.update_layout(height=max(400, len(bow_df.index) * 40))
            st.plotly_chart(fig_bow_heatmap, use_container_width=True)
            
            # Word frequency across documents
            word_totals = bow_df.sum().sort_values(ascending=False)
            fig_word_freq = px.bar(x=word_totals.index, y=word_totals.values,
                                  title="Word Frequency Across All Documents",
                                  labels={'x': 'Words', 'y': 'Total Count'})
            st.plotly_chart(fig_word_freq, use_container_width=True)
            
            # Comparison with sklearn
            st.subheader("🔬 Comparison with Scikit-learn")
            
            vectorizer = CountVectorizer(lowercase=True, token_pattern=r'\b[a-zA-Z]+\b')
            sklearn_bow = vectorizer.fit_transform(documents)
            sklearn_vocab = vectorizer.get_feature_names_out()
            
            sklearn_df = pd.DataFrame(sklearn_bow.toarray(), 
                                     columns=sklearn_vocab,
                                     index=[f'Doc {i+1}' for i in range(len(documents))])
            
            st.write("**Scikit-learn BoW Matrix:**")
            st.dataframe(sklearn_df)
            
            # BoW limitations
            st.subheader("⚠️ Bag of Words Limitations")
            st.markdown("""
            1. **No word order**: "cat sat mat" vs "mat sat cat" are identical
            2. **No context**: "bank" (financial) vs "bank" (river) are the same
            3. **Sparse vectors**: Most values are zero (inefficient)
            4. **No semantic similarity**: "king" and "queen" are completely different
            5. **Vocabulary size**: Vector size grows with vocabulary (can be huge)
            """)
    
    # Tab 2: Word2Vec
    with tab2:
        st.header("🧠 Word2Vec")
        
        st.markdown("""
        ## What is Word2Vec?
        
        **Word2Vec** creates dense vector representations of words that capture semantic meaning. Words with similar meanings have similar vectors.
        
        ### Key Ideas:
        - **Distributional Hypothesis**: Words that appear in similar contexts have similar meanings
        - **Dense Vectors**: Each word is represented by a vector of real numbers (typically 100-300 dimensions)
        - **Semantic Relationships**: Vector arithmetic captures relationships (king - man + woman ≈ queen)
        
        ### Two Architectures:
        1. **CBOW (Continuous Bag of Words)**: Predict target word from context words
        2. **Skip-gram**: Predict context words from target word
        
        ### Training Process:
        1. **Sliding Window**: Move a window across text to create word pairs
        2. **Neural Network**: Train a shallow neural network to predict word relationships
        3. **Vector Extraction**: Use the learned weights as word vectors
        """)
        
        if documents and len(documents) >= 3:
            st.divider()
            st.subheader("🔧 Training Word2Vec Model")
            
            # Prepare data for Word2Vec
            sentences = [simple_preprocess(doc) for doc in documents]
            
            # Show preprocessing
            st.write("**1. Text Preprocessing for Word2Vec**")
            for i, (original, processed) in enumerate(zip(documents, sentences)):
                with st.expander(f"Document {i+1}"):
                    st.write(f"**Original**: {original}")
                    st.write(f"**Processed**: {processed}")
            
            # Train Word2Vec model
            st.write("**2. Training Word2Vec Model**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                vector_size = st.slider("Vector Size", 10, 100, 50)
            with col2:
                window_size = st.slider("Window Size", 1, 10, 5)
            with col3:
                min_count = st.slider("Min Count", 1, 3, 1)
            
            try:
                # Train model
                model = Word2Vec(sentences, 
                               vector_size=vector_size, 
                               window=window_size, 
                               min_count=min_count, 
                               workers=1,
                               seed=42)
                
                # Get vocabulary
                vocab = list(model.wv.key_to_index.keys())
                
                st.success(f"✅ Model trained successfully! Vocabulary size: {len(vocab)}")
                
                # Show word vectors
                st.subheader("📊 Word Vectors")
                
                selected_words = st.multiselect("Select words to examine:", 
                                               vocab, 
                                               default=vocab[:min(5, len(vocab))])
                
                if selected_words:
                    # Create vectors dataframe
                    vectors_data = []
                    for word in selected_words:
                        vector = model.wv[word]
                        vectors_data.append([word] + vector.tolist())
                    
                    columns = ['Word'] + [f'Dim_{i+1}' for i in range(vector_size)]
                    vectors_df = pd.DataFrame(vectors_data, columns=columns)
                    
                    st.dataframe(vectors_df.style.format({'Word': '{}', **{f'Dim_{i+1}': '{:.3f}' for i in range(vector_size)}}))
                
                # Word similarities
                st.subheader("🔍 Word Similarities")
                
                if len(vocab) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        word1 = st.selectbox("Select first word:", vocab, key="word1")
                    with col2:
                        word2 = st.selectbox("Select second word:", vocab, key="word2")
                    
                    if word1 and word2 and word1 != word2:
                        similarity = model.wv.similarity(word1, word2)
                        st.metric("Cosine Similarity", f"{similarity:.4f}")
                        
                        # Show most similar words
                        if st.button("Find most similar words"):
                            try:
                                similar_words = model.wv.most_similar(word1, topn=min(5, len(vocab)-1))
                                st.write(f"**Words most similar to '{word1}':**")
                                for word, sim in similar_words:
                                    st.write(f"• {word}: {sim:.4f}")
                            except:
                                st.warning("Not enough data to find similar words.")
                
                # Vector visualization
                st.subheader("📈 Vector Visualization")
                
                if len(vocab) >= 3:
                    # Dimensionality reduction
                    reduction_method = st.radio("Reduction method:", ["PCA", "t-SNE"])
                    
                    # Get all word vectors
                    word_vectors = np.array([model.wv[word] for word in vocab])
                    
                    if reduction_method == "PCA":
                        reducer = PCA(n_components=2, random_state=42)
                    else:
                        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(vocab)-1))
                    
                    reduced_vectors = reducer.fit_transform(word_vectors)
                    
                    # Create scatter plot
                    fig_scatter = go.Figure()
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=reduced_vectors[:, 0],
                        y=reduced_vectors[:, 1],
                        mode='markers+text',
                        text=vocab,
                        textposition='top center',
                        marker=dict(size=10, color='blue'),
                        name='Words'
                    ))
                    
                    fig_scatter.update_layout(
                        title=f"Word2Vec Visualization ({reduction_method})",
                        xaxis_title=f"{reduction_method} Component 1",
                        yaxis_title=f"{reduction_method} Component 2",
                        height=500
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Vector arithmetic
                st.subheader("🧮 Vector Arithmetic")
                st.markdown("""
                One of the most fascinating properties of Word2Vec is that vector arithmetic often captures semantic relationships:
                - king - man + woman ≈ queen
                - Paris - France + Italy ≈ Rome
                """)
                
                if len(vocab) >= 3:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        word_a = st.selectbox("Word A:", vocab, key="arith_a")
                    with col2:
                        word_b = st.selectbox("Word B (subtract):", vocab, key="arith_b")
                    with col3:
                        word_c = st.selectbox("Word C (add):", vocab, key="arith_c")
                    
                    if word_a and word_b and word_c and len(set([word_a, word_b, word_c])) == 3:
                        if st.button("Calculate: A - B + C"):
                            try:
                                result = model.wv.most_similar(positive=[word_a, word_c], 
                                                             negative=[word_b], 
                                                             topn=3)
                                st.write(f"**{word_a} - {word_b} + {word_c} ≈**")
                                for word, similarity in result:
                                    st.write(f"• {word} (similarity: {similarity:.4f})")
                            except:
                                st.warning("Not enough data for vector arithmetic.")
                
            except Exception as e:
                st.error(f"Error training Word2Vec model: {str(e)}")
                st.info("Try using more documents or reducing the minimum word count.")
        
        else:
            st.warning("Please provide at least 3 documents to train a Word2Vec model.")
    
    # Tab 3: Comparison
    with tab3:
        st.header("🔄 Bag of Words vs Word2Vec")
        
        # Comparison table
        comparison_data = {
            "Aspect": [
                "Representation",
                "Vector Size",
                "Semantic Meaning",
                "Word Order",
                "Context",
                "Sparsity",
                "Training Required",
                "Memory Usage",
                "Computational Cost",
                "Interpretability"
            ],
            "Bag of Words": [
                "Sparse, count-based",
                "Size of vocabulary",
                "No semantic understanding",
                "Completely ignored",
                "No context awareness",
                "Very sparse (mostly zeros)",
                "No training needed",
                "High (large vocabulary)",
                "Low",
                "High (direct word counts)"
            ],
            "Word2Vec": [
                "Dense, real-valued",
                "Fixed (50-300 dimensions)",
                "Captures semantic relationships",
                "Indirectly captured through context",
                "Context-aware",
                "Dense (no zeros)",
                "Requires training on corpus",
                "Low (fixed dimensions)",
                "High (neural network training)",
                "Low (abstract representations)"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        st.divider()
        
        # When to use what
        st.subheader("🎯 When to Use Each Method")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Use Bag of Words when:
            - **Simple tasks**: Basic text classification
            - **Small datasets**: Limited training data
            - **Interpretability**: Need to understand feature importance
            - **Baseline models**: Quick prototyping
            - **Keyword matching**: Exact word matching is important
            """)
        
        with col2:
            st.markdown("""
            ### Use Word2Vec when:
            - **Semantic understanding**: Need meaning-based similarity
            - **Large datasets**: Sufficient data for training
            - **Efficiency**: Memory and computation constraints
            - **Advanced NLP**: Building sophisticated models
            - **Transfer learning**: Pre-trained embeddings available
            """)
        
        st.divider()
        
        # Evolution to modern embeddings
        st.subheader("🚀 Evolution to Modern Embeddings")
        
        st.markdown("""
        ### The Journey Continues...
        
        **Word2Vec** was revolutionary, but the field has evolved further:
        
        1. **FastText** (2016): Handles out-of-vocabulary words using subword information
        2. **GloVe** (2014): Combines global matrix factorization with local context windows
        3. **ELMo** (2018): Context-dependent embeddings (same word, different meanings)
        4. **BERT** (2018): Bidirectional, transformer-based, context-aware
        5. **GPT** (2018+): Generative, transformer-based, context-aware
        6. **Modern LLMs**: ChatGPT, Claude, etc. - sophisticated understanding
        
        ### Key Improvements:
        - **Context-dependent**: Same word can have different embeddings based on context
        - **Bidirectional**: Consider both left and right context
        - **Transfer learning**: Pre-trained on massive corpora
        - **Attention mechanisms**: Focus on relevant parts of input
        - **Multilingual**: Work across different languages
        """)
        
        # Learning path
        st.subheader("📚 Your Learning Path")
        
        st.markdown("""
        ### Recommended Next Steps:
        
        1. **Practice with real data**: Try these methods on your own text datasets
        2. **Explore pre-trained embeddings**: Use Word2Vec models trained on large corpora
        3. **Learn about attention**: Understand how transformers work
        4. **Experiment with BERT**: Try Hugging Face transformers library
        5. **Build applications**: Create text classification, similarity search, or recommendation systems
        
        ### Key Takeaways:
        - **Start simple**: Bag of Words is still useful for many tasks
        - **Understand the progression**: Each method builds on previous insights
        - **Context matters**: Modern embeddings capture context better
        - **No one-size-fits-all**: Choose the right tool for your specific task
        """)

if __name__ == "__main__":
    show_bow_word2vec_app()

