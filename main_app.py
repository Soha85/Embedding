import streamlit as st
from tfidf_app import show_tfidf_app
from bow_word2vec_app import show_bow_word2vec_app

# Page configuration
st.set_page_config(
    page_title="Text Embedding Education",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def show_home():
    st.title("📚 Text Embedding Education")
    st.markdown("### Understanding the Journey from Words to Vectors")
    
    # Hero section
    st.markdown("""
    Welcome to an interactive learning experience that will take you through the evolution of text representation in Natural Language Processing (NLP). 
    From simple word counting to sophisticated neural embeddings, discover how computers learn to understand human language.
    """)
    
    st.divider()
    
    # Learning path overview
    st.header("🗺️ Your Learning Journey")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎒 1. Bag of Words
        **The Foundation**
        - Simple word counting
        - Vocabulary creation
        - Sparse representations
        - Easy to understand
        
        *Start here to understand the basics*
        """)
    
    with col2:
        st.markdown("""
        ### 📊 2. TF-IDF
        **Adding Intelligence**
        - Term frequency analysis
        - Document importance weighting
        - Better than simple counting
        - Still interpretable
        
        *Learn how to weight word importance*
        """)
    
    with col3:
        st.markdown("""
        ### 🧠 3. Word2Vec
        **Semantic Understanding**
        - Dense vector representations
        - Captures word relationships
        - Semantic similarity
        - Foundation for modern NLP
        
        *Discover how meaning is captured*
        """)
    
    st.divider()
    
    # Why this matters
    st.header("🎯 Why This Matters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Real-World Applications
        - **Search Engines**: How Google understands your queries
        - **Recommendation Systems**: Netflix, Spotify, Amazon suggestions
        - **Chatbots**: Customer service automation
        - **Content Analysis**: Social media monitoring
        - **Translation**: Google Translate and similar services
        - **Sentiment Analysis**: Understanding emotions in text
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Career Relevance
        - **Data Science**: Essential for text analytics
        - **Machine Learning**: Foundation for NLP models
        - **Software Engineering**: Building intelligent applications
        - **Product Management**: Understanding AI capabilities
        - **Research**: Academic and industry research
        - **Entrepreneurship**: Building AI-powered startups
        """)
    
    st.divider()
    
    # Learning objectives
    st.header("🎓 What You'll Learn")
    
    objectives = [
        "**Understand the evolution** of text representation methods",
        "**Implement algorithms** from scratch to see how they work",
        "**Compare different approaches** and understand their trade-offs",
        "**Visualize concepts** through interactive charts and examples",
        "**Apply knowledge** to real-world text data",
        "**Prepare for advanced topics** like BERT, GPT, and modern transformers"
    ]
    
    for obj in objectives:
        st.markdown(f"✅ {obj}")
    
    st.divider()
    
    # Getting started
    st.header("🚀 Getting Started")
    
    st.markdown("""
    ### Recommended Learning Path:
    
    1. **Start with Bag of Words** - Understand the fundamentals of converting text to numbers
    2. **Move to TF-IDF** - Learn how to weight word importance intelligently  
    3. **Explore Word2Vec** - Discover how semantic meaning can be captured in vectors
    4. **Compare all methods** - Understand when to use each approach
    
    ### Tips for Success:
    - 📝 **Experiment with your own text** - Try the examples with your own documents
    - 🔍 **Pay attention to visualizations** - They reveal important patterns
    - 🤔 **Think about limitations** - Understanding weaknesses is as important as strengths
    - 🔗 **Connect concepts** - See how each method builds on the previous one
    - 💡 **Ask questions** - Use the interactive features to explore edge cases
    """)
    
    # Call to action
    st.markdown("""
    ---
    ### Ready to begin? Choose a topic from the sidebar to start your journey! 🎯
    """)

def main():
    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    st.sidebar.markdown("Choose a topic to explore:")
    
    # Navigation options
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🎒 Bag of Words & Word2Vec", "📊 TF-IDF"],
        index=0
    )
    
    # Add some spacing
    st.sidebar.markdown("---")
    
    # Learning progress tracker
    st.sidebar.markdown("### 📈 Learning Progress")
    
    # Simple progress tracking using session state
    if 'visited_pages' not in st.session_state:
        st.session_state.visited_pages = set()
    
    # Add current page to visited
    if page != "🏠 Home":
        st.session_state.visited_pages.add(page)
    
    # Show progress
    total_topics = 2  # BoW/Word2Vec and TF-IDF
    completed = len(st.session_state.visited_pages)
    progress = completed / total_topics
    
    st.sidebar.progress(progress)
    st.sidebar.write(f"Topics explored: {completed}/{total_topics}")
    
    # Show visited topics
    if st.session_state.visited_pages:
        st.sidebar.write("✅ Completed:")
        for visited in st.session_state.visited_pages:
            st.sidebar.write(f"  • {visited}")
    
    st.sidebar.markdown("---")
    
    # Additional resources
    st.sidebar.markdown("""
    ### 📖 Additional Resources
    
    **Books:**
    - "Speech and Language Processing" by Jurafsky & Martin
    - "Natural Language Processing with Python" by Bird, Klein & Loper
    
    **Online Courses:**
    - CS224N: Natural Language Processing with Deep Learning (Stanford)
    - Natural Language Processing Specialization (Coursera)
    
    **Papers:**
    - "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)
    - "Distributed Representations of Words and Phrases" (Word2Vec improvements)
    """)
    
    # Main content area
    if page == "🏠 Home":
        show_home()
    elif page == "🎒 Bag of Words & Word2Vec":
        show_bow_word2vec_app()
    elif page == "📊 TF-IDF":
        show_tfidf_app()

if __name__ == "__main__":
    main()

