"""
FORENSIC CLUSTERING PIPELINE (SIMPLIFIED)
Focus: Clustering only, no ordering
Status: Test with 4 file types, 10 fragments each

Pipeline:
  1. LOAD: 10 fragments each of 4 file types (40 total)
  2. FEATURES: Byte frequency (256-dim) + N-grams (128-dim)
  3. SIMILARITY: Cosine similarity
  4. CLUSTERING: DBSCAN to group by file type
  5. EVALUATION: ARI, NMI, Purity scores
"""

import os
import sys
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random

sys.path.insert(0, os.path.dirname(__file__))
from fragment_ordering_optimized import parse_fragment_name, get_files_for_type


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_byte_frequency(fragment_data):
    """Extract 256-dimensional byte frequency vector"""
    vector = np.zeros(256, dtype=np.float32)
    for byte in fragment_data:
        vector[byte] += 1
    
    if np.sum(vector) > 0:
        vector = vector / np.sum(vector)
    
    return vector


def extract_ngrams(fragment_data, n=4, max_features=128):
    """Extract top-128 n-gram frequencies"""
    ngrams = []
    for i in range(len(fragment_data) - n + 1):
        ngram = tuple(fragment_data[i:i+n])
        ngrams.append(ngram)
    
    if not ngrams:
        return np.zeros(max_features, dtype=np.float32)
    
    ngram_counts = Counter(ngrams)
    top_ngrams = [ngram for ngram, _ in ngram_counts.most_common(max_features)]
    
    vector = np.zeros(max_features, dtype=np.float32)
    for idx, ngram in enumerate(top_ngrams):
        vector[idx] = ngram_counts[ngram]
    
    if np.sum(vector) > 0:
        vector = vector / np.sum(vector)
    
    return vector


def extract_features(fragment_data):
    """Combine byte frequency + n-grams into single feature vector"""
    byte_freq = extract_byte_frequency(fragment_data)
    ngrams = extract_ngrams(fragment_data, n=4, max_features=128)
    
    # Combine: 60% byte frequency + 40% n-grams
    combined = np.concatenate([
        byte_freq * 0.6,
        ngrams * 0.4
    ])
    
    return combined


# ============================================================================
# CLUSTERING
# ============================================================================

def cluster_with_dbscan(features, eps=0.15, min_samples=2):
    """DBSCAN clustering on feature space"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(features)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    return labels, n_clusters, n_noise


def cluster_with_kmeans(features, n_clusters=4):
    """K-Means clustering with fixed number of clusters"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return labels, n_clusters


# ============================================================================
# EVALUATION
# ============================================================================

def compute_purity(true_labels, predicted_labels):
    """Compute cluster purity"""
    n_samples = len(true_labels)
    
    cluster_true = defaultdict(list)
    for pred, true in zip(predicted_labels, true_labels):
        cluster_true[pred].append(true)
    
    purity_sum = 0
    for cluster_id, true_list in cluster_true.items():
        if len(true_list) > 0:
            purity_sum += max(Counter(true_list).values())
    
    return purity_sum / n_samples


def evaluate_clustering(predicted_labels, true_labels):
    """Compute clustering metrics: ARI, NMI, Purity"""
    
    # Handle noise points (-1 label)
    pred_for_eval = predicted_labels.copy()
    if -1 in pred_for_eval:
        pred_for_eval[pred_for_eval == -1] = max(predicted_labels) + 1
    
    ari = adjusted_rand_score(true_labels, pred_for_eval)
    nmi = normalized_mutual_info_score(true_labels, pred_for_eval)
    purity = compute_purity(true_labels, pred_for_eval)
    
    return {
        'ARI': ari,
        'NMI': nmi,
        'Purity': purity
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_fragments(dataset_root, file_types, frags_per_type):
    """
    Load equal number of fragments from each file type
    
    Args:
        dataset_root: path to Train/ folder
        file_types: list of file types
        frags_per_type: how many fragments per type
    
    Returns:
        fragments: {filename: bytes}
        true_labels: {filename: file_type_index}
        file_list: {file_type: [filenames]}
    """
    fragments = {}
    true_labels = {}
    file_list = {}
    
    print(f"\nLoading {frags_per_type} fragments each from {len(file_types)} file types...")
    
    for type_idx, ftype in enumerate(file_types):
        files = get_files_for_type(dataset_root, ftype)
        
        # Shuffle and take first N
        random.shuffle(files)
        files = files[:frags_per_type]
        
        file_list[ftype] = []
        
        for fpath in files:
            fname = os.path.basename(fpath)
            
            try:
                with open(fpath, 'rb') as f:
                    data = f.read()
                
                if len(data) > 0:
                    fragments[fname] = data
                    true_labels[fname] = type_idx
                    file_list[ftype].append(fname)
            except:
                pass
        
        print(f"  [{type_idx+1}] {ftype.upper()}: {len(file_list[ftype])} fragments loaded")
    
    return fragments, true_labels, file_list


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run clustering-only pipeline with 3 file types (HTML, PDF, APK)"""
    
    dataset_root = "Train"
    file_types = ["html", "pdf", "apk"]  # 3 file types ONLY
    frags_per_type = 10  # 10 each
    
    print("\n" + "="*100)
    print("FORENSIC CLUSTERING PIPELINE (SIMPLIFIED)")
    print("="*100)
    
    # ===== STEP 1: LOAD =====
    print("\n[STEP 1] LOAD FRAGMENTS")
    print("-" * 100)
    
    fragments, true_labels, file_list = load_fragments(
        dataset_root, file_types, frags_per_type
    )
    
    total_frags = len(fragments)
    print(f"\nTotal fragments loaded: {total_frags}")
    
    # ===== STEP 2: FEATURE EXTRACTION =====
    print("\n[STEP 2] FEATURE EXTRACTION")
    print("-" * 100)
    
    fragment_names = sorted(fragments.keys())
    
    print(f"Extracting features from {len(fragment_names)} fragments...")
    print(f"  Feature type: Byte Frequency (256-dim) + N-grams (128-dim)")
    print(f"  Combination: 60% byte frequency + 40% n-grams")
    
    features = []
    for fname in fragment_names:
        feat = extract_features(fragments[fname])
        features.append(feat)
    
    features = np.array(features)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Total feature dimension: {features.shape[1]}")
    
    # ===== STEP 3: CLUSTERING =====
    print("\n[STEP 3] CLUSTERING (UNSUPERVISED)")
    print("-" * 100)
    
    # Use K-Means with k=3 (3 file types: HTML, PDF, APK)
    print("\nUsing K-Means Clustering (k=3 - one cluster per file type)...")
    print("  Algorithm: K-Means with fixed random seed for reproducible results")
    
    # Single K-Means run with fixed seed for exact results
    labels, _ = cluster_with_kmeans(features, n_clusters=3)
    
    # Evaluate
    true_labels_array = np.array([true_labels[fname] for fname in fragment_names])
    pred_for_eval = labels.copy()
    purity = compute_purity(true_labels_array, pred_for_eval)
    
    n_clusters = 3
    algorithm_used = f"K-Means (k=3)"
    
    print(f"  Result: Purity = {purity:.1%}")
    print("-"*100)
    
    # ===== STEP 4: RESULTS =====
    print("\n[STEP 4] CLUSTERING RESULTS")
    print("-" * 100)
    
    # Group fragments by cluster
    clusters = defaultdict(list)
    for fname, label in zip(fragment_names, labels):
        clusters[label].append(fname)
    
    print(f"\nAlgorithm used: {algorithm_used}")
    print(f"Number of clusters: {n_clusters}\n")
    
    # Show each cluster
    for cluster_id in sorted(clusters.keys()):
        cluster_frags = clusters[cluster_id]
        
        # Count file types in this cluster
        type_counts = defaultdict(int)
        for fname in cluster_frags:
            ftype_idx = true_labels[fname]
            ftype = file_types[ftype_idx]
            type_counts[ftype] += 1
        
        type_summary = ", ".join([f"{t}:{type_counts[t]}" 
                                  for t in sorted(type_counts.keys())])
        
        print(f"CLUSTER {cluster_id} ({len(cluster_frags)} fragments):")
        print(f"  Composition: {type_summary}")
        print(f"  Fragments:")
        
        for fname in sorted(cluster_frags):
            ftype = file_types[true_labels[fname]]
            print(f"    - {fname} ({ftype})")
        
        print()
    
    # ===== STEP 5: EVALUATION =====
    print("[STEP 5] EVALUATION METRICS")
    print("-" * 100)
    
    # Convert true labels to array
    true_labels_array = np.array([true_labels[fname] for fname in fragment_names])
    
    # Compute metrics
    metrics = evaluate_clustering(labels, true_labels_array)
    
    print(f"\nMetrics (comparing predicted clusters to true file types):")
    print(f"  ✓ Purity:                  {metrics['Purity']:.2%}")
    print(f"    (Fraction of correct clusters)")
    print(f"\n  ✓ ARI (Adjusted Rand Index): {metrics['ARI']:.4f}")
    print(f"    (1.0 = perfect, 0.0 = random, -1.0 = worse than random)")
    print(f"\n  ✓ NMI (Normalized Mutual Info): {metrics['NMI']:.4f}")
    print(f"    (1.0 = perfect, 0.0 = independent)")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    print(f"""
Configuration:
  File types: {', '.join(file_types)} ({len(file_types)} types)
  Fragments per type: {frags_per_type}
  Total fragments: {total_frags}
  Features: 256-dim byte freq + 128-dim n-grams = 384-dim total
  Algorithm: {algorithm_used}

Results:
  Clusters identified: {n_clusters}
  Clustering Purity: {metrics['Purity']:.1%}
  ARI: {metrics['ARI']:.4f}
  NMI: {metrics['NMI']:.4f}
""")
    print("="*100)


if __name__ == "__main__":
    main()
