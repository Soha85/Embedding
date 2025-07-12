## Phase 1: Plan and design the educational app series
- [ ] Outline content and features for TF-IDF app
- [ ] Outline content and features for Word2Vec and Bag of Words app
- [ ] Plan how to connect the apps

## Phase 2: Create TF-IDF educational Streamlit app
- [x] Create comprehensive TF-IDF educational app with step-by-step explanations
- [x] Include manual calculations and visualizations
- [x] Add comparison with scikit-learn implementation

## Phase 3: Create Word2Vec and Bag of Words educational app
- [x] Create comprehensive BoW and Word2Vec educational app
- [x] Include step-by-step explanations for both methods
- [x] Add interactive visualizations and comparisons
- [x] Show evolution to modern embeddings

## Phase 4: Create main navigation app to connect all educational modules
- [x] Create main navigation app with home page
- [x] Integrate all educational modules
- [x] Add progress tracking and learning path guidance
- [x] Create requirements.txt and README.md files

## Phase 5: Test and deploy the educational app series
- [x] Test main navigation app locally
- [x] Test TF-IDF educational module
- [x] Test Bag of Words & Word2Vec module
- [x] Verify all interactive features work correctly
- [x] Confirm progress tracking functionality

## Phase 6: Deliver the completed educational applications to user


- [ ] Outline content and features for TF-IDF app
  - [ ] Allow user to input multiple text documents.
  - [ ] Display term frequency (TF) for each document.
  - [ ] Display inverse document frequency (IDF) for each term.
  - [ ] Display TF-IDF scores for each term in each document.
  - [ ] Provide a clear explanation of TF, IDF, and TF-IDF concepts.
  - [ ] Visualize the TF-IDF matrix.



- [ ] Outline content and features for Word2Vec and Bag of Words app
  - [ ] Bag of Words:
    - [ ] Allow user to input text documents.
    - [ ] Display the vocabulary.
    - [ ] Display the Bag of Words representation for each document.
    - [ ] Explain the concept of Bag of Words.
  - [ ] Word2Vec:
    - [ ] Allow user to input text for training a Word2Vec model.
    - [ ] Display word embeddings (vectors) for selected words.
    - [ ] Visualize word similarities (e.g., using t-SNE or PCA).
    - [ ] Explain the concept of Word2Vec and its training process.



- [ ] Plan how to connect the apps
  - [ ] Create a main Streamlit app that serves as a navigation hub.
  - [ ] Each individual app (TF-IDF, BoW/Word2Vec) will be a separate Python file.
  - [ ] The main app will use `st.sidebar` or `st.radio` to switch between the different modules.


