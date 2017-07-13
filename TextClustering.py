###############################################################################
# Name: Jinkal Arvind Javia                                     
# Text Clustering
###############################################################################

import string
import re
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# read the input data file 'foods.txt' and Extracting review/text part of input file
input_file = open("foods.txt","r")
extracted_reviews = []
for i in input_file:
	temp = i.split(":")	
	if temp[0] == "review/text":
		extracted_reviews.append(temp[1])	
input_file.close()

# read the input data file 'Long Stopword List.txt'
long_stopword_list = []
input_file = open("Long Stopword List.txt", "r")
for i in input_file:
	long_stopword_list.append(re.sub("\n","",i))
input_file.close()

# Perform preprocessing steps on extracted_reviews
for i in range(len(extracted_reviews)):
	extracted_reviews[i] = extracted_reviews[i].lower()
	extracted_reviews[i] = extracted_reviews[i].strip()
	extracted_reviews[i] = re.sub('<br />', ' ', extracted_reviews[i])
	extracted_reviews[i] = re.sub(r'\d', ' ', extracted_reviews[i])
	punctuations = re.compile('[^a-zA-Z]+')
	extracted_reviews[i] = punctuations.sub(' ', extracted_reviews[i])

# Extract unique words from extracted_reviews and store them L
vect = CountVectorizer()
matrix = vect.fit_transform(extracted_reviews)
vocab = list(vect.get_feature_names())
L = []
for i in vocab:
	L.append(str(i))

# Remove stopwords from L and store the result in W
vect = CountVectorizer(stop_words = long_stopword_list)
matrix = vect.fit_transform(extracted_reviews)
vocab = list(vect.get_feature_names())
W = []
for i in vocab:
	W.append(str(i))

# Count word frequency and extract 500 most frequent words
top_500_list = []
counts = matrix.sum(axis = 0).A1
freq_dist = Counter(dict(zip(vocab, counts)))
top_500 = freq_dist.most_common(500)

# Store Top 500 words along with their count to output file 'top_500.txt'
input_file = open("top_500.txt", "w")
for i in top_500:
    temp = i
    top_500_list.append(str(list(temp)[0]))
    input_file.write(str(i))
    input_file.write("\n")
input_file.close()

# Perform vectorization using Term Frequency Inverse Document Frequency
tfidf = TfidfVectorizer(vocabulary = top_500_list)
transform = tfidf.fit_transform(extracted_reviews)

# Perform K-Means clustering where k = 10
kmeans = KMeans(n_clusters = 10, max_iter = 10).fit(transform)

# Extract Top 5 words representing each cluster and their feature value
# Write them to output file 'CentroidTopWords.txt'
input_file = open("CentroidTopWords.txt", "w")
cluster_centroid = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = top_500_list
for i in range(10):
	for j in cluster_centroid[i, :5]:
		input_file.write(terms[j])
		input_file.write(" ")
	input_file.write("\n")
input_file.close()
