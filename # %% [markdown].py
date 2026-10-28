# %% [markdown]
# ## Information Retrieval Component: Individual Project 
# 
# Mia Busuttil- 0150406L
# 
# ###Introduction
# In this project we are required to index and query a WES data set. The project is divided into 2 sepearte parts the Indexing and the Querying. 
# 
# ###Part 1: Indexing
# I created a python code box for importing all the necessary libraries. 
# To load and read the data in XMLs raw tags I decided to use REGEX. This is the abbrivation of regualr expression.l I chose this as it is a process I am familiar with. I ouputted the number of documnets extracted to maintain engineering hygeine:
# Extracted 331 documents
# 
# I then tokenized the document and output a sample to ensure the code was logically correct:
# Document 1 tokens: ['<', '!', '[', 'CDATA', '[', 'Joseph', 'Nicollet', 'and', 'the', 'Upper', 'Mississippi', 'River', '.', 'On', 'July', '24', ',', '1786', ',', 'French']
# Document 2 tokens: ['<', '!', '[', 'CDATA', '[', 'Beaumarchais', 'and', 'Figaro', "'s", 'Wedding', '.', 'On', 'May', '18', ',', '1799', ',', 'French', 'playwright', 'Pierre-Augustin']
# Preprocessing done
# Original document: <![CDATA[Joseph Nicollet and the Upper Mississippi River.
# 
# Lastly I built the terms by document matrix containing the TF.IDF weight for eahc term within the document
# On July 24, 1786, French geographer, astr ...
# Preprocessed tokens: ['cdata', 'joseph', 'nicollet', 'upper', 'mississippi', 'river', 'juli', '24', '1786', 'french', 'geograph', 'astronom', 'mathematician', 'joseph', 'nicola', 'nicollet', 'born', 'best', 'known', 'map'] ...
#    aachen  abadon   abandon  abbasid  abbevil  abbey  abbot  abbé  abc  \
# 0     0.0     0.0  0.000000      0.0      0.0    0.0    0.0   0.0  0.0   
# 1     0.0     0.0  0.024862      0.0      0.0    0.0    0.0   0.0  0.0   
# 2     0.0     0.0  0.000000      0.0      0.0    0.0    0.0   0.0  0.0   
# 3     0.0     0.0  0.000000      0.0      0.0    0.0    0.0   0.0  0.0   
# 4     0.0     0.0  0.000000      0.0      0.0    0.0    0.0   0.0  0.0   
# 
#    abdomin  
# 0      0.0  
# 1      0.0  
# 2      0.0  
# 3      0.0  
# 4      0.0  
# And that is Part 1 completed. 
# 
# ###Part 2: Querying
# I set the query to "French geographer goes missing in the alps" I then preprocessed the query by passing it through the tokenization, case-folding, stop-word removal and stemming models created in the indexing part of the code. This was the succesful output: 
# French geographer goes missing in Alps
# Original query: French geographer goes missing in Alps
# Preprocessed query tokens: ['french', 'geograph', 'goe', 'miss', 'alp']
# 
# I then used cosine similarity to calculate the similarity between the query i created and each document. This was the result:
# Top 5 relevant documents
# Document 204 with similarity score 0.0792
# <![CDATA[Louis Antoine de Bougainville and his Voyage Around the World.
# 
# Bougainville reaching Tahiti Probably on November 11, 1729, French admiral and explorer Louis Antoine de Bougainville was born. ...
# 
# 
# Document 326 with similarity score 0.0664
# <![CDATA[Claude Joseph Rouget de Lisle and the Marseillaise.
# 
# Claude Joseph Rouget de Lisle singing La Marseillaise.  On April 25, 1792, French army officier Claude Joseph Rouget de Lisle during the F ...
# 
# 
# Document 192 with similarity score 0.0571
# <![CDATA[John Tyndall and the Physics of Air.
# 
# John Tyndall (1820-1893). caricatured as a preacher in the magazine Vanity Fair, 1872.  On August 2, 1820, British physicist John Tyndall was born. His i ...
# 
# 
# Document 250 with similarity score 0.0532
# <![CDATA[Abraham Ortelius and the Theatrum Orbis Terrarum.
# ...
# 
# Erich von Drygalski (1865-1949).  .  On February 9, 1865, German geographer, geophysicist and polar scientist Erich Dagobert von Drygalski was bo ...
# 
# I then outputted the top 5 relevant documents:
# Ranked list of documents based on query similarity:
# 
# 1. Document 204 (similarity score: 0.0792)
# <![CDATA[Louis Antoine de Bougainville and his Voyage Around the World.
# 
# Bougainville reaching Tahiti Probably on November 11, 1729, French admiral and explorer Louis Antoine de Bougainville was born. ...
# 
# 2. Document 326 (similarity score: 0.0664)
# <![CDATA[Claude Joseph Rouget de Lisle and the Marseillaise.
# 
# Claude Joseph Rouget de Lisle singing La Marseillaise.  On April 25, 1792, French army officier Claude Joseph Rouget de Lisle during the F ...
# 
# 3. Document 192 (similarity score: 0.0571)
# <![CDATA[John Tyndall and the Physics of Air.
# 
# John Tyndall (1820-1893). caricatured as a preacher in the magazine Vanity Fair, 1872.  On August 2, 1820, British physicist John Tyndall was born. His i ...
# 
# 4. Document 250 (similarity score: 0.0532)
# <![CDATA[Abraham Ortelius and the Theatrum Orbis Terrarum.
# 
# Ortelius World Map – Typus Orbis Terrarum 1570.  On May 20, 1570, Belgian cartographer and geographer Abraham Ortelius publishes the first m ...
# 
# 5. Document 113 (similarity score: 0.0528)
# <![CDATA[Erich von Drygalski’s Antarctic Expeditions.
# 
# Erich von Drygalski (1865-1949).  .  On February 9, 1865, German geographer, geophysicist and polar scientist Erich Dagobert von Drygalski was bo ...
# 
# ###Conclusion\
# Overall this assignment was simple and strait to the point. The results from my code were as expected and there are minimal logical errors. 

# %%
#importing 
import nltk
import os
import re
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Imports done")

# %%
#PART 1: INDEXING 
#first load the data and read the docoment
folder_path = 'docs'

all_docs = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
 
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
            
            
            matches = [m.strip() for m in re.findall(r'<raw>(.*?)</raw>', data, re.DOTALL | re.IGNORECASE)]
            all_docs.extend(matches)

print(f"Extracted {len(all_docs)} documents")

#tokenisation
nltk.download('punkt')

tokenize_docs= [word_tokenize(doc) for doc in all_docs]

#output first 2 tokenised documents
for i, tokens in enumerate(tokenize_docs[:2], 1):
    print(f"Document {i} tokens: {tokens[:20]}")  # show first 20 tokens

#extraction of index terms 
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

preprocessed_docs = []

for doc in all_docs:
    doc_lower=doc.lower()
    tokens = nltk.word_tokenize(doc_lower)
    tokens_nostop = [word for word in tokens if word.isalnum() and word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in tokens_nostop]
    preprocessed_docs.append(stemmed_tokens)
print("Preprocessing done")

#example
print("Original document:", all_docs[0][:100], "...")
print("Preprocessed tokens:", preprocessed_docs[0][:20], "...")

#weighting of index terms 
filtered_docs = [[token for token in doc if token.isalpha()] for doc in preprocessed_docs]
docs_as_strings = [' '.join(doc) for doc in preprocessed_docs] 
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x,      
    preprocessor=lambda x: x,   
    token_pattern=None           
)
tfidf_matrix = vectorizer.fit_transform(filtered_docs) 
terms = vectorizer.get_feature_names_out() 
tdm_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)
print(tdm_df.iloc[:5, :10])

# %%
#Part 2: Querying
#create query
query= "French geographer goes missing in Alps"
print (query)

#preprocess query
query_lower = query.lower()
query_tokens = nltk.word_tokenize(query_lower)
query_nostop = [word for word in query_tokens if word.isalnum() and word not in stop_words]
query_stemmed = [stemmer.stem(word) for word in query_nostop]
print("Original query:", query)
print("Preprocessed query tokens:", query_stemmed)

#csine similarity between query and docs
query_vec = vectorizer.transform([query_stemmed])
similarity=cosine_similarity(query_vec, tfidf_matrix).flatten()
ranked_indices = similarity.argsort()[::-1]
print("Top 5 relevant documents:")
for idx in ranked_indices[:5]:
    print(f"Document {idx} with similarity score {similarity[idx]:.4f}")
    print(all_docs[idx][:200], "...\n")
    print()

#printing top 5 documents on cosine similiarity 
print("Ranked list of documents based on query similarity:\n")
for rank, idx in enumerate(ranked_indices, 1):
    print(f"{rank}. Document {idx} (similarity score: {similarity[idx]:.4f})")


