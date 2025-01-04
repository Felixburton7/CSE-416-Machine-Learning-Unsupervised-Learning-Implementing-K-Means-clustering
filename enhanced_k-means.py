# Clustering Text Data with k-means

# Hey there! 😊 This project is all about using the k-means algorithm to cluster some Wikipedia text documents.
# Instead of relying on sklearn's implementation, we're building k-means from scratch with NumPy.
# Let's dive in and see what insights we can uncover from these unlabeled documents!

# Importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Enable inline plotting for Jupyter notebooks
%matplotlib inline

# Loading the data
wiki = pd.read_csv('people_wiki.csv')
print("Here's a sneak peek at the data:")
print(wiki.head(20))

# Converting text data into numerical features using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Setting max_df to 0.95 to ignore very common words
vectorizer = TfidfVectorizer(max_df=0.95)
tf_idf = vectorizer.fit_transform(wiki['text'])
words = vectorizer.get_feature_names_out()
print("\nSome of the words we're working with:")
print(words)

# Normalizing the TF-IDF vectors to unit length
from sklearn.preprocessing import normalize
tf_idf = normalize(tf_idf)
print("\nTF-IDF vectors have been normalized.")

# Function to initialize centroids by randomly selecting k data points
def get_initial_centroids(data, k, seed=None):
    if seed is not None:
        np.random.seed(seed)  # For reproducibility
    n = data.shape[0]
    rand_indices = np.random.choice(n, k, replace=False)
    centroids = data[rand_indices, :].toarray()
    print(f"Initialized centroids with seed={seed}")
    return centroids

# Function to assign each data point to the nearest centroid
from sklearn.metrics import pairwise_distances

def assign_clusters(data, centroids):
    distances = pairwise_distances(data, centroids, metric='euclidean')
    closest_cluster = np.argmin(distances, axis=1)
    return closest_cluster

# Function to update centroids based on current cluster assignments
def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
        member_data_points = data[cluster_assignment == i]
        if member_data_points.shape[0] > 0:
            centroid = member_data_points.mean(axis=0)
            new_centroids.append(centroid)
        else:
            # If a cluster gets no points, reinitialize its centroid randomly
            random_idx = np.random.choice(data.shape[0])
            centroid = data[random_idx, :].toarray()
            new_centroids.append(centroid)
            print(f"Cluster {i} had no points. Reinitialized its centroid.")
    new_centroids = np.array(new_centroids)
    return new_centroids

# Function to compute heterogeneity (sum of squared distances)
def compute_heterogeneity(data, k, centroids, cluster_assignment):
    heterogeneity = 0.0
    for i in range(k):
        member_data_points = data[cluster_assignment == i]
        if member_data_points.shape[0] > 0:
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances ** 2
            heterogeneity += np.sum(squared_distances)
    return heterogeneity

# The main k-means algorithm
def kmeans(data, k, initial_centroids, max_iter=100, record_heterogeneity=None, verbose=False):
    centroids = initial_centroids.copy()
    prev_cluster_assignment = None
    
    for itr in range(max_iter):
        if verbose:
            print(f'\nIteration {itr + 1}')
        
        # Assign clusters
        cluster_assignment = assign_clusters(data, centroids)
        
        # Update centroids
        centroids = revise_centroids(data, k, cluster_assignment)
        
        # Check for convergence (no change in assignments)
        if prev_cluster_assignment is not None and (prev_cluster_assignment == cluster_assignment).all():
            if verbose:
                print("Convergence reached! No change in cluster assignments.")
            break
        
        # Record heterogeneity if needed
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)
            if verbose:
                print(f"Heterogeneity: {score:.4f}")
        
        prev_cluster_assignment = cluster_assignment.copy()
    
    return centroids, cluster_assignment

# k-means++ initialization for better centroid selection
def k_means_plus_plus_initialization(data, k, seed=None):
    if seed is not None:
        np.random.seed(seed)  # For reproducibility
    
    centroids = []
    # Randomly choose the first centroid
    idx = np.random.randint(data.shape[0])
    centroids.append(data[idx, :].toarray()[0])
    
    for _ in range(1, k):
        distances = pairwise_distances(data, np.array(centroids), metric='euclidean')
        min_distances = distances.min(axis=1)
        probs = min_distances / min_distances.sum()
        next_idx = np.random.choice(data.shape[0], p=probs)
        centroids.append(data[next_idx, :].toarray()[0])
        print(f"Selected new centroid at index {next_idx}")
    
    return np.array(centroids)

# Enhanced k-means with tolerance-based convergence
def enhanced_kmeans(data, k, initial_centroids, max_iter=100, tolerance=1e-4, record_heterogeneity=None, verbose=False):
    centroids = initial_centroids.copy()
    prev_heterogeneity = float('inf')
    
    for itr in range(max_iter):
        if verbose:
            print(f'\nIteration {itr + 1}')
        
        # Assign clusters
        cluster_assignment = assign_clusters(data, centroids)
        
        # Update centroids
        centroids = revise_centroids(data, k, cluster_assignment)
        
        # Compute current heterogeneity
        current_heterogeneity = compute_heterogeneity(data, k, centroids, cluster_assignment)
        
        if verbose:
            print(f"Heterogeneity: {current_heterogeneity:.4f}")
        
        # Check for convergence based on tolerance
        if prev_heterogeneity - current_heterogeneity < tolerance:
            if verbose:
                print("Convergence reached based on heterogeneity improvement.")
            break
        
        if record_heterogeneity is not None:
            record_heterogeneity.append(current_heterogeneity)
        
        prev_heterogeneity = current_heterogeneity
    
    return centroids, cluster_assignment

# Function to plot heterogeneity over iterations
def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(8, 5))
    plt.plot(heterogeneity, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Heterogeneity')
    plt.title(f'Heterogeneity Over Iterations (K={k})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Running enhanced k-means with k=3 clusters
k = 3
heterogeneity = []
initial_centroids = k_means_plus_plus_initialization(tf_idf, k, seed=0)
centroids, cluster_assignment = enhanced_kmeans(
    tf_idf, 
    k, 
    initial_centroids, 
    max_iter=400, 
    tolerance=1e-4, 
    record_heterogeneity=heterogeneity, 
    verbose=True
)
plot_heterogeneity(heterogeneity, k)

# Identifying the largest cluster
largest_cluster = np.bincount(cluster_assignment).argmax()
print(f'\n🎉 The largest cluster is Cluster {largest_cluster} with {np.bincount(cluster_assignment)[largest_cluster]} documents.')

# Function to run k-means multiple times with different seeds and select the best result
def kmeans_multiple_runs(data, k, max_iter=100, seeds=10, tolerance=1e-4, verbose=False):
    min_heterogeneity = float('inf')
    best_centroids = None
    best_assignment = None
    
    for seed in seeds:
        if verbose:
            print(f'\nRunning k-means with seed={seed}')
        initial_centroids = k_means_plus_plus_initialization(data, k, seed=seed)
        centroids, cluster_assignment = enhanced_kmeans(
            data, 
            k, 
            initial_centroids, 
            max_iter=max_iter, 
            tolerance=tolerance, 
            verbose=verbose
        )
        current_heterogeneity = compute_heterogeneity(data, k, centroids, cluster_assignment)
        print(f"Heterogeneity for seed {seed}: {current_heterogeneity:.4f}")
        
        if current_heterogeneity < min_heterogeneity:
            min_heterogeneity = current_heterogeneity
            best_centroids = centroids
            best_assignment = cluster_assignment
    
    return best_centroids, best_assignment

# Function to plot heterogeneity vs. number of clusters
def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, heterogeneity_values, marker='s', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Running k-means for different values of k to find the optimal number of clusters
all_centroids = {}
all_cluster_assignment = {}
heterogeneity_values = []
seeds = [20000, 40000, 80000]
k_list = [2, 10, 25, 50, 100]

for k_val in k_list:
    print(f'\n🚀 Running k-means for K={k_val}')
    centroids_k, assignment_k = kmeans_multiple_runs(
        tf_idf, 
        k=k_val, 
        max_iter=400, 
        seeds=seeds, 
        tolerance=1e-4, 
        verbose=True
    )
    all_centroids[k_val] = centroids_k
    all_cluster_assignment[k_val] = assignment_k
    score = compute_heterogeneity(tf_idf, k_val, centroids_k, assignment_k)
    heterogeneity_values.append(score)

# Plotting heterogeneity vs. k
plot_k_vs_heterogeneity(k_list, heterogeneity_values)

# Function to visualize document clusters
def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, words, display_docs=5):
    print('=' * 90)
    
    for c in range(k):
        num_docs = (cluster_assignment == c).sum()
        print(f'🔹 Cluster {c} ({num_docs} documents)')
        
        # Get top words in the centroid
        top_indices = centroids[c].argsort()[::-1][:5]
        top_words = [f"{words[idx]}:{centroids[c, idx]:.3f}" for idx in top_indices]
        print("Top words:", ' | '.join(top_words))
        
        if display_docs > 0:
            print("\n📄 Top documents in this cluster:")
            distances = pairwise_distances(tf_idf, centroids[c].reshape(1, -1), metric='euclidean').flatten()
            distances[cluster_assignment != c] = float('inf')
            nearest_neighbors = distances.argsort()[:display_docs]
            
            for i in nearest_neighbors:
                doc_name = wiki.iloc[i]['name']
                doc_distance = distances[i]
                doc_text = ' '.join(wiki.iloc[i]['text'].split()[:25])  # First 25 words
                print(f"* {doc_name:50s} (Distance: {doc_distance:.5f})")
                print(f"  {doc_text}...\n")
        
        print('=' * 90)

# Visualizing clusters for k=2
k = 2
print(f'\n🔍 Visualizing clusters for K={k}')
visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words)

# Visualizing clusters for k=10
k = 10
print(f'\n🔍 Visualizing clusters for K={k}')
visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words)

# Counting small clusters for k=100
k = 100
num_small_clusters = (np.bincount(all_cluster_assignment[k]) < 44).sum()
print(f'\n📊 Number of small clusters (fewer than 44 articles) when K={k}: {num_small_clusters}')
