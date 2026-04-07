"""
DIGITAL FORENSICS: FRAGMENT RECONSTRUCTION PIPELINE
Research-Level Implementation

Pipeline:
  1. INPUT: Load fragments (unknown)
  2. FEATURE EXTRACTION: Byte frequency + N-grams
  3. SIMILARITY: Cosine similarity matrix
  4. CLUSTERING: DBSCAN (unsupervised)
  5. ORDERING: Greedy nearest-neighbor within clusters
  6. EVALUATION: ARI, purity, accuracy (SEPARATE from logic)

Key Principle: NO ground truth used during clustering/ordering
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
# STEP 1: FEATURE EXTRACTION
# ============================================================================

def extract_byte_frequency(fragment_data, normalize=True):
    """
    Extract 256-dimensional byte frequency vector from fragment
    
    Args:
        fragment_data: bytes object
        normalize: if True, normalize to sum=1
    
    Returns:
        np.array of shape (256,) with byte frequencies
    """
    vector = np.zeros(256, dtype=np.float32)
    
    for byte in fragment_data:
        vector[byte] += 1
    
    if normalize and np.sum(vector) > 0:
        vector = vector / np.sum(vector)
    
    return vector


def extract_entropy(fragment_data):
    """Extract entropy-based and pattern-based features"""
    if len(fragment_data) == 0:
        return np.zeros(16, dtype=np.float32)
    
    # Byte frequency for entropy calculation
    byte_freq = np.zeros(256)
    for byte in fragment_data:
        byte_freq[byte] += 1
    
    byte_freq_nonzero = byte_freq[byte_freq > 0]
    byte_prob = byte_freq_nonzero / len(fragment_data)
    
    # Shannon entropy
    entropy = -np.sum(byte_prob * np.log2(byte_prob + 1e-10))
    
    # Chi-square against uniform distribution
    expected = len(fragment_data) / 256
    chi_square = np.sum(((byte_freq - expected) ** 2) / (expected + 1e-10))
    
    # Byte range (how many different bytes are present)
    unique_bytes = np.count_nonzero(byte_freq) / 256.0
    
    # Pattern detection
    null_bytes = (fragment_data.count(b'\x00')) / len(fragment_data)
    high_entropy_ratio = np.sum(byte_freq[[0xFF, 0xFE, 0xFD, 0xFC]]) / len(fragment_data)
    ascii_printable = np.sum(byte_freq[32:127]) / len(fragment_data)
    
    # Byte transitions (adjacent bytes that are different)
    transitions = 0
    if len(fragment_data) > 1:
        for i in range(len(fragment_data) - 1):
            if fragment_data[i] != fragment_data[i+1]:
                transitions += 1
        transition_ratio = transitions / (len(fragment_data) - 1)
    else:
        transition_ratio = 0
    
    # Repeated bytes (compression indicator)
    max_run = 1
    current_run = 1
    for i in range(1, len(fragment_data)):
        if fragment_data[i] == fragment_data[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    repeat_ratio = max_run / len(fragment_data)
    
    # Create feature vector (16 dims)
    features = np.array([
        entropy / 8.0,
        chi_square / 10000.0,
        unique_bytes,
        np.mean(byte_freq) / len(fragment_data),
        np.std(byte_freq) / len(fragment_data),
        np.max(byte_freq) / len(fragment_data),
        np.count_nonzero(byte_freq[0:32]) / 32.0,  # Control chars
        ascii_printable,  # ASCII printable
        np.count_nonzero(byte_freq[127:256]) / 129.0,  # High bytes
        null_bytes,
        high_entropy_ratio,
        transition_ratio,
        repeat_ratio,
        np.min(byte_freq[byte_freq > 0]) / len(fragment_data) if len(byte_freq_nonzero) > 0 else 0,
        np.median(byte_freq[byte_freq > 0]) / len(fragment_data) if len(byte_freq_nonzero) > 0 else 0,
        np.percentile(byte_freq[byte_freq > 0], 75) / len(fragment_data) if len(byte_freq_nonzero) > 0 else 0,
    ], dtype=np.float32)
    
    return features


def extract_ngrams(fragment_data, n=4, max_features=128):
    """
    Extract n-gram features from fragment
    
    Args:
        fragment_data: bytes object
        n: n-gram size (default 4)
        max_features: limit number of unique n-grams to track (for efficiency)
    
    Returns:
        np.array of shape (max_features,) with n-gram frequencies (normalized)
    """
    # Extract all n-grams as tuples
    ngrams = []
    for i in range(len(fragment_data) - n + 1):
        ngram = tuple(fragment_data[i:i+n])
        ngrams.append(ngram)
    
    if not ngrams:
        return np.zeros(max_features, dtype=np.float32)
    
    # Count n-gram frequencies
    ngram_counts = Counter(ngrams)
    
    # Get top max_features n-grams
    top_ngrams = [ngram for ngram, _ in ngram_counts.most_common(max_features)]
    
    # Create feature vector
    vector = np.zeros(max_features, dtype=np.float32)
    for idx, ngram in enumerate(top_ngrams):
        vector[idx] = ngram_counts[ngram]
    
    # Normalize
    if np.sum(vector) > 0:
        vector = vector / np.sum(vector)
    
    return vector


def combine_features(byte_freq, ngrams, weights=None):
    """
    Combine byte frequency and n-gram features
    
    Args:
        byte_freq: np.array of shape (256,)
        ngrams: np.array of shape (max_features,)
        weights: tuple (w_byte, w_ngram) or None for equal weight
    
    Returns:
        np.array with concatenated features
    """
    if weights is None:
        weights = (0.6, 0.4)  # Byte frequency more important
    
    w_byte, w_ngram = weights
    
    # Normalize weights
    total_w = w_byte + w_ngram
    w_byte = w_byte / total_w
    w_ngram = w_ngram / total_w
    
    # Combine: concatenate weighted features
    combined = np.concatenate([
        byte_freq * w_byte,
        ngrams * w_ngram
    ])
    
    return combined


# ============================================================================
# STEP 2: SIMILARITY COMPUTATION
# ============================================================================

def compute_suffix_prefix_overlap(fragments_dict, fragment_names, window_size=256):
    """
    Compute overlap similarity based on suffix-prefix matching
    Key for fragment ordering: adjacent fragments should have matching boundaries
    
    Args:
        fragments_dict: {filename: bytes}
        fragment_names: list of filenames
        window_size: size of suffix/prefix to check (default 256)
    
    Returns:
        overlap_matrix: (N, N) matrix of overlap scores
    """
    n = len(fragment_names)
    overlap_matrix = np.zeros((n, n))
    
    for i, fname_i in enumerate(fragment_names):
        frag_i = fragments_dict[fname_i]
        
        # Suffix of fragment i
        suffix_i = frag_i[-window_size:] if len(frag_i) >= window_size else frag_i
        
        for j, fname_j in enumerate(fragment_names):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue
            
            frag_j = fragments_dict[fname_j]
            
            # Prefix of fragment j
            prefix_j = frag_j[:window_size] if len(frag_j) >= window_size else frag_j
            
            # Compute overlap: how much do they match?
            overlap_len = min(len(suffix_i), len(prefix_j))
            if overlap_len == 0:
                overlap_matrix[i, j] = 0.0
                continue
            
            # Count matching bytes
            matches = sum(1 for k in range(overlap_len) if suffix_i[k] == prefix_j[k])
            overlap_matrix[i, j] = matches / overlap_len
    
    return overlap_matrix


def compute_similarity_matrix(fragments_dict, fragment_names):
    """
    Compute combined similarity matrix: feature-based + overlap-based
    
    Args:
        fragments_dict: {filename: bytes}
        fragment_names: list of filenames
    
    Returns:
        similarity_matrix: np.array of shape (N, N) with combined similarities
        features: np.array of shape (N, feature_dim) with extracted features
    """
    print("\n[FEATURE EXTRACTION]")
    print(f"  Extracting features from {len(fragment_names)} fragments...")
    
    features = []
    for fname in fragment_names:
        byte_freq = extract_byte_frequency(fragments_dict[fname])
        entropy_feat = extract_entropy(fragments_dict[fname])
        ngrams = extract_ngrams(fragments_dict[fname], n=4)
        combined = np.concatenate([
            byte_freq * 0.35,
            entropy_feat * 0.35,
            ngrams * 0.30
        ])
        features.append(combined)
    
    features = np.array(features)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Feature dimension: {features.shape[1]}")
    
    print("\n[SIMILARITY COMPUTATION]")
    print(f"  Computing feature-based similarity...")
    feature_sim = cosine_similarity(features)
    
    print(f"  Computing suffix-prefix overlap similarity...")
    overlap_sim = compute_suffix_prefix_overlap(fragments_dict, fragment_names, window_size=256)
    
    # Combine: 60% feature similarity + 40% overlap (overlap is better for ordering)
    similarity_matrix = 0.6 * feature_sim + 0.4 * overlap_sim
    
    print(f"  Combined similarity matrix shape: {similarity_matrix.shape}")
    print(f"  Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    
    return similarity_matrix, features


# ============================================================================
# STEP 3: CLUSTERING (UNSUPERVISED - NO GROUND TRUTH)
# ============================================================================
def cluster_fragments_kmeans(features, n_clusters=2, verbose=True):
    """
    Cluster fragments using K-Means (fallback for DBSCAN)
    
    Args:
        features: feature matrix of shape (N, feature_dim)
        n_clusters: number of clusters
        verbose: if True, print clustering info
    
    Returns:
        labels: cluster labels
        n_clusters: number of clusters
    """
    if verbose:
        print("\n[K-MEANS CLUSTERING (FALLBACK)]")
        print(f"  Parameters: k={n_clusters}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    if verbose:
        cluster_counts = defaultdict(int)
        for label in labels:
            cluster_counts[label] += 1
        
        print(f"  Cluster distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            print(f"    Cluster {cluster_id}: {cluster_counts[cluster_id]} fragments")
    
    return labels, n_clusters

def cluster_fragments_dbscan(similarity_matrix, fragment_names, eps=0.20, min_samples=2, verbose=True, features=None):
    """
    Cluster fragments using DBSCAN
    
    IMPORTANT: This function uses NO ground truth. It only uses similarity/features.
    
    Args:
        similarity_matrix: np.array of shape (N, N) with values in [-1, 1]
        fragment_names: list of filenames
        eps: density parameter 
        min_samples: minimum samples per cluster
        verbose: if True, print clustering info
        features: feature matrix for Euclidean distance calculation
    
    Returns:
        labels: cluster labels (-1 = noise)
        n_clusters: number of clusters found
    """
    if verbose:
        print("\n[DBSCAN CLUSTERING]")
        print(f"  Parameters: eps={eps}, min_samples={min_samples}")
    
    if features is not None:
        # Use Euclidean distance on feature space
        from sklearn.metrics.pairwise import euclidean_distances
        distance_matrix = euclidean_distances(features)
        if verbose:
            print(f"  Using Euclidean distance on feature space")
            print(f"  Distance range: [{distance_matrix.min():.4f}, {distance_matrix.max():.4f}]")
    else:
        # Convert similarity to distance: distance = 1 - similarity
        distance_matrix = 1.0 - similarity_matrix
        distance_matrix = np.clip(distance_matrix, 0, None)
    
    # DBSCAN on distance matrix
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if verbose:
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        # Print cluster distribution
        cluster_counts = defaultdict(int)
        for label in labels:
            cluster_counts[label] += 1
        
        print(f"  Cluster distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            if cluster_id == -1:
                print(f"    Noise (-1): {cluster_counts[cluster_id]} fragments")
            else:
                print(f"    Cluster {cluster_id}: {cluster_counts[cluster_id]} fragments")
    
    return labels, n_clusters


def group_by_clusters(fragment_names, labels):
    """
    Group fragments by their predicted cluster labels
    
    Args:
        fragment_names: list of filenames
        labels: cluster labels from DBSCAN
    
    Returns:
        clusters: {cluster_id: [fragment_names]}
    """
    clusters = defaultdict(list)
    
    for name, label in zip(fragment_names, labels):
        if label != -1:  # Ignore noise points
            clusters[label].append(name)
    
    return clusters


# ============================================================================
# STEP 4: ORDERING WITHIN CLUSTERS
# ============================================================================

def order_fragments_in_cluster(cluster_fragments, fragments_dict, similarity_matrix, 
                                fragment_names, method='multiple_optimized'):
    """
    Order fragments within a cluster using optimized algorithms
    
    Args:
        cluster_fragments: list of fragment names in cluster
        fragments_dict: {filename: bytes}
        similarity_matrix: full similarity matrix (NxN)
        fragment_names: list of all fragment names (for indexing)
        method: 'greedy', '2opt', or 'multiple'
    
    Returns:
        ordered: list of fragment names in optimal order
    """
    if len(cluster_fragments) <= 1:
        return cluster_fragments
    
    # Get indices of cluster fragments in full matrix
    indices = [fragment_names.index(fname) for fname in cluster_fragments]
    
    # Extract sub-matrix for this cluster
    cluster_sim = similarity_matrix[np.ix_(indices, indices)]
    
    # Convert similarity to distance (minimize)
    cluster_dist = 1.0 - cluster_sim  # Invert: high similarity = low distance
    
    if method == '2opt':
        # Start with greedy nearest-neighbor, then optimize with 2-opt
        greedy_order = _greedy_nn_order(cluster_dist)
        optimized_order = _two_opt_optimize(greedy_order, cluster_dist)
        
        # Always rotate to start at index 0
        zero_idx = optimized_order.index(0)
        optimized_order = optimized_order[zero_idx:] + optimized_order[:zero_idx]
        
        ordered = [cluster_fragments[i] for i in optimized_order]
        
    elif method == 'multiple' or method == 'multiple_optimized':
        # Try multiple random starts and pick best
        best_order = None
        best_cost = float('inf')
        
        num_starts = 25 if method == 'multiple_optimized' else 10
        
        for _ in range(num_starts):  #More random starts for better solution
            random_order = list(range(len(cluster_fragments)))
            random.shuffle(random_order)
            optimized = _two_opt_optimize(random_order, cluster_dist, max_iterations=5000)
            
            # Try all rotations and pick best cost
            for rotation_idx in range(len(optimized)):
                rotated = optimized[rotation_idx:] + optimized[:rotation_idx]
                cost = _calculate_path_cost(rotated, cluster_dist)
                if cost < best_cost:
                    best_cost = cost
                    best_order = rotated
        
        # Make sure it starts at low index
        if best_order and 0 in best_order:
            zero_idx = best_order.index(0)
            best_order = best_order[zero_idx:] + best_order[:zero_idx]
        
        ordered = [cluster_fragments[i] for i in best_order] if best_order else cluster_fragments
        
    else:  # greedy
        greedy_order = _greedy_nn_order(cluster_dist)
        ordered = [cluster_fragments[i] for i in greedy_order]
    
    return ordered


def _greedy_nn_order(distance_matrix):
    """Greedy nearest-neighbor to build initial tour"""
    n = len(distance_matrix)
    unvisited = set(range(n))
    current = 0
    order = [0]
    unvisited.remove(0)
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
        order.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return order


def _two_opt_optimize(initial_order, distance_matrix, max_iterations=1000):
    """
    2-opt optimization for TSP-like problem
    Iteratively swap edges to minimize total distance
    """
    order = list(initial_order)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                if j == len(order) - 1 and i == 0:
                    continue  # Skip if it would just reverse the whole tour
                
                # Calculate current cost of edges
                a, b = order[i], order[i + 1]
                c, d = order[j], order[(j + 1) % len(order)]
                
                current_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                
                # If swap improves, do it
                if new_cost < current_cost:
                    # Reverse the segment between i+1 and j
                    order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                    improved = True
                    break
            
            if improved:
                break
    
    return order


def _calculate_path_cost(order, distance_matrix):
    """Calculate total cost of a path"""
    cost = 0
    for i in range(len(order)):
        a = order[i]
        b = order[(i + 1) % len(order)]
        cost += distance_matrix[a, b]
    return cost



# ============================================================================
# STEP 5: EVALUATION (SEPARATE - USES GROUND TRUTH ONLY FOR METRICS)
# ============================================================================

def evaluate_clustering(predicted_labels, true_labels, metric_names=None):
    """
    Evaluate clustering quality using ground truth (for validation ONLY)
    
    IMPORTANT: This function is SEPARATE from clustering logic
    It only computes metrics, does not affect the pipeline
    
    Args:
        predicted_labels: cluster labels from DBSCAN
        true_labels: ground truth labels (file types or indices)
        metric_names: list of metrics to compute
    
    Returns:
        metrics: {metric_name: value}
    """
    if metric_names is None:
        metric_names = ['ari', 'nmi', 'purity']
    
    metrics = {}
    
    # Adjust labels for noise (-1)
    pred_for_eval = predicted_labels.copy()
    pred_for_eval[pred_for_eval == -1] = max(predicted_labels) + 1
    
    if 'ari' in metric_names:
        ari = adjusted_rand_score(true_labels, pred_for_eval)
        metrics['ari'] = ari
    
    if 'nmi' in metric_names:
        nmi = normalized_mutual_info_score(true_labels, pred_for_eval)
        metrics['nmi'] = nmi
    
    if 'purity' in metric_names:
        purity = compute_purity(true_labels, pred_for_eval)
        metrics['purity'] = purity
    
    return metrics


def compute_purity(true_labels, predicted_labels):
    """
    Compute cluster purity (how pure are predicted clusters)
    
    Args:
        true_labels: ground truth labels
        predicted_labels: predicted cluster labels
    
    Returns:
        purity: value between 0 and 1
    """
    n_samples = len(true_labels)
    
    cluster_true = defaultdict(list)
    for pred, true in zip(predicted_labels, true_labels):
        cluster_true[pred].append(true)
    
    purity_sum = 0
    for cluster_id, true_list in cluster_true.items():
        if len(true_list) > 0:
            purity_sum += max(Counter(true_list).values())
    
    return purity_sum / n_samples


def evaluate_ordering_within_cluster(ordered_fragments, true_indices):
    """
    Evaluate ordering accuracy within a cluster using multiple metrics
    
    Args:
        ordered_fragments: list of fragment names in order
        true_indices: {fragment_name: (file_type, original_index)}
    
    Returns:
        metrics: dict with 'consecutive_pairs', 'lis_ratio', 'inversions', 'spearman_corr'
    """
    if len(ordered_fragments) <= 1:
        return {'consecutive_pairs': 1.0, 'lis_ratio': 1.0, 'inversions': 0, 'spearman_corr': 1.0}
    
    # Extract original indices
    true_seq = []
    reconstructed_seq = []
    for i, fname in enumerate(ordered_fragments):
        if fname in true_indices:
            _, true_idx = true_indices[fname]
            true_seq.append(true_idx)
            reconstructed_seq.append(i)  # position in reconstructed order
    
    # Metric 1: Consecutive pairs (what we had before)
    correct_pairs = 0
    total_pairs = 0
    for i in range(len(ordered_fragments) - 1):
        curr_name = ordered_fragments[i]
        next_name = ordered_fragments[i + 1]
        if curr_name in true_indices and next_name in true_indices:
            _, curr_idx = true_indices[curr_name]
            _, next_idx = true_indices[next_name]
            if curr_idx + 1 == next_idx:
                correct_pairs += 1
            total_pairs += 1
    
    consecutive_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    
    # Metric 2: Longest Increasing Subsequence (how many fragments are in correct relative order)
    if true_seq:
        lis_length = len(compute_lis(true_seq))
        lis_ratio = lis_length / len(true_seq)
    else:
        lis_ratio = 0.0
    
    # Metric 3: Inversion count (number of pairs in wrong order)
    inversions = 0
    for i in range(len(true_seq)):
        for j in range(i + 1, len(true_seq)):
            if true_seq[i] > true_seq[j]:  # Out of order
                inversions += 1
    
    max_inversions = len(true_seq) * (len(true_seq) - 1) // 2
    inversion_penalty = inversions / max_inversions if max_inversions > 0 else 0
    
    # Metric 4: Spearman correlation (overall rank correlation)
    if len(true_seq) > 1:
        # Compute rank-based correlation
        expected = list(range(len(true_seq)))
        actual = sorted(range(len(true_seq)), key=lambda i: true_seq[i])
        
        # Simple correlation: how many are in correct order
        correlation = 1 - (2 * inversions / max_inversions if max_inversions > 0 else 0)
    else:
        correlation = 1.0
    
    return {
        'consecutive_pairs': consecutive_accuracy,
        'lis_ratio': lis_ratio,
        'inversions': inversions,
        'spearman_corr': correlation
    }


def compute_lis(arr):
    """Compute Longest Increasing Subsequence indices"""
    if not arr:
        return []
    
    n = len(arr)
    dp = [1] * n
    parent = [-1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # Build LIS
    max_len = max(dp)
    max_idx = dp.index(max_len)
    
    lis = []
    idx = max_idx
    while idx != -1:
        lis.append(arr[idx])
        idx = parent[idx]
    
    return list(reversed(lis))


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_fragments(dataset_root, file_types, case_id, frags_per_type):
    """
    Load fragments from dataset
    
    Args:
        dataset_root: path to Train/ folder
        file_types: list of file types to load
        case_id: case ID (e.g., "0001")
        frags_per_type: fragments per type
    
    Returns:
        fragments_dict: {filename: bytes}
        true_indices: {filename: (file_type, original_idx)} for evaluation
        file_list: {file_type: [filenames]} for display
    """
    fragments_dict = {}
    true_indices = {}
    file_list = {}
    
    for file_type in file_types:
        files = get_files_for_type(dataset_root, file_type)
        
        # Group by case ID
        groups = defaultdict(list)
        for fpath in files:
            cid, fidx, ftype = parse_fragment_name(fpath)
            if cid:
                groups[cid].append((fidx, fpath))
        
        if case_id not in groups:
            continue
        
        frag_list = groups[case_id]
        frag_list.sort(key=lambda x: x[0])
        frag_list = frag_list[:frags_per_type]
        
        file_list[file_type] = []
        
        for idx, fpath in frag_list:
            name = os.path.basename(fpath)
            with open(fpath, "rb") as f:
                fragments_dict[name] = f.read()
            
            true_indices[name] = (file_type, idx)
            file_list[file_type].append(name)
    
    return fragments_dict, true_indices, file_list


def main(dataset_root="Train", file_types=None, case_id="0001", frags_per_type=10,
         eps=0.20, min_samples=2, show_evaluation=True):
    """
    Run complete forensic pipeline
    
    Args:
        dataset_root: path to dataset
        file_types: list of file types
        case_id: case ID
        frags_per_type: fragments per type
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples
        show_evaluation: if True, show evaluation metrics (requires ground truth)
    """
    if file_types is None:
        file_types = ["html", "pdf"]
    
    print("\n" + "="*100)
    print("FORENSIC PIPELINE: Fragment Reconstruction")
    print("="*100)
    
    # ===== LOAD =====
    print("\n[STEP 1] LOAD FRAGMENTS")
    print("-" * 100)
    
    fragments_dict, true_indices, file_list = load_fragments(
        dataset_root, file_types, case_id, frags_per_type
    )
    
    n_fragments = len(fragments_dict)
    print(f"\nLoaded {n_fragments} fragments:")
    
    for ftype in file_types:
        if ftype in file_list:
            files = file_list[ftype]
            print(f"\n  {ftype.upper()}: {len(files)} fragments")
            for i, fname in enumerate(files, 1):
                size = len(fragments_dict[fname])
                print(f"    {i}. {fname} ({size} bytes)")
    
    # ===== SHUFFLE (simulate unknown order) =====
    print("\n[STEP 2] SHUFFLE FRAGMENTS (Simulate Unknown Order)")
    print("-" * 100)
    
    fragment_names = list(fragments_dict.keys())
    random.shuffle(fragment_names)
    
    print(f"All {len(fragment_names)} fragments shuffled to unknown order")
    
    # ===== FEATURE EXTRACTION & SIMILARITY =====
    print("\n[STEP 3] FEATURE EXTRACTION & SIMILARITY")
    print("-" * 100)
    
    similarity_matrix, features = compute_similarity_matrix(fragments_dict, fragment_names)
    
    # ===== CLUSTERING =====
    print("\n[STEP 4] CLUSTERING (UNSUPERVISED)")
    print("-" * 100)
    
    labels, n_clusters = cluster_fragments_dbscan(
        similarity_matrix, fragment_names, eps=0.020, min_samples=3, features=features
    )
    
    # Fallback to K-Means if DBSCAN creates only 1 cluster
    if n_clusters <= 1:
        print("\n[Fallback: DBSCAN produced single cluster, using K-Means with k=2]")
        labels, n_clusters = cluster_fragments_kmeans(features, n_clusters=2)
    
    clusters = group_by_clusters(fragment_names, labels)
    
    # ===== ORDERING =====
    print("\n[STEP 5] ORDERING WITHIN CLUSTERS")
    print("-" * 100)
    
    ordered_clusters = {}
    for cluster_id in sorted(clusters.keys()):
        cluster_frags = clusters[cluster_id]
        
        ordered = order_fragments_in_cluster(
            cluster_frags, fragments_dict, similarity_matrix, fragment_names
        )
        
        ordered_clusters[cluster_id] = ordered
        
        # Determine cluster type
        cluster_type = true_indices[ordered[0]][0] if ordered else "unknown"
        
        print(f"\n[CLUSTER {cluster_id}] {cluster_type.upper()} - {len(ordered)} fragments:")
        print("-" * 100)
        
        # Count fragment types in cluster
        type_counts = defaultdict(int)
        for fname in ordered:
            ftype, _ = true_indices[fname]
            type_counts[ftype] += 1
        type_summary = ', '.join([f"{t}:{type_counts[t]}" for t in sorted(type_counts.keys())])
        print(f"Composition: {type_summary}\n")
        
        for i, fname in enumerate(ordered):
            ftype, orig_idx = true_indices[fname]
            
            # Check if this is correct order (consecutive index from same type)
            status = ""
            if i > 0:
                prev_fname = ordered[i-1]
                prev_ftype, prev_idx = true_indices[prev_fname]
                
                if prev_ftype == ftype and prev_idx + 1 == orig_idx:
                    status = " [CORRECT]"
                elif prev_ftype != ftype:
                    status = " [TYPE CHANGE]"
                else:
                    status = " [WRONG]"
            
            print(f"  {i+1:2d}. {fname:40s} (idx:{orig_idx}) {status}")
        
        # === ORDER ACCURACY DETAILS ===
        correct_pairs = 0
        total_transitions = 0
        print(f"\n  SEQUENTIAL ORDER ANALYSIS:")
        print(f"  {'-'*96}")
        
        # Show reconstructed sequence
        reconstructed_indices = [true_indices[fname][1] for fname in ordered]
        original_indices = list(range(len(ordered)))  # What they should be
        original_indices.sort(key=lambda i: reconstructed_indices[i])
        
        print(f"    Original order should be:    {list(range(len(ordered)))}")
        print(f"    Your reconstruction got:     {reconstructed_indices}")
        print()
        
        for i in range(len(ordered) - 1):
            curr_fname = ordered[i]
            next_fname = ordered[i + 1]
            curr_ftype, curr_idx = true_indices[curr_fname]
            next_ftype, next_idx = true_indices[next_fname]
            
            total_transitions += 1
            
            # Check transition
            if curr_ftype == next_ftype and curr_idx + 1 == next_idx:
                marker = "[CORRECT]"
                correct_pairs += 1
            elif curr_ftype != next_ftype:
                marker = "[TYPE CHANGE]"
            else:
                marker = "[SKIP]"
            
            print(f"    {curr_idx} -> {next_idx}: {curr_fname[:30]:30s} -> {next_fname[:30]:30s} {marker}")
        
        # Calculate comprehensive metrics
        metrics = evaluate_ordering_within_cluster(ordered, true_indices)
        
        if total_transitions > 0:
            accuracy = (correct_pairs / total_transitions) * 100
            print(f"\n  CONSECUTIVE PAIRS: {correct_pairs}/{total_transitions} = {accuracy:.1f}%")
        else:
            print(f"\n  CONSECUTIVE PAIRS: Single fragment (N/A)")
        
        # Show additional metrics
        print(f"  LIS RATIO (% in correct relative order): {metrics['lis_ratio']:.1%}")
        print(f"  INVERSIONS (# of wrong pairs): {metrics['inversions']}")
        print(f"  OVERALL CORRELATION: {metrics['spearman_corr']:.1%}")
    
    # ===== EVALUATION (SEPARATE - USE GROUND TRUTH ONLY FOR METRICS) =====
    if show_evaluation:
        print("\n[STEP 6] EVALUATION METRICS")
        print("-" * 100)
        
        # Convert true file types to numeric labels for clustering evaluation
        true_labels = np.array([0 if true_indices[fname][0] == file_types[0] else 1 
                                for fname in fragment_names])
        
        metrics = evaluate_clustering(labels, true_labels)
        
        print(f"\nClustering Quality:")
        print(f"  Purity: {metrics['purity']:.2%}")
        print(f"  ARI (Adjusted Rand Index): {metrics['ari']:.4f}")
        print(f"  NMI (Normalized Mutual Info): {metrics['nmi']:.4f}")
        
        # Ordering accuracy
        print(f"\nOrdering Quality Summary:")
        for cluster_id in sorted(ordered_clusters.keys()):
            metrics = evaluate_ordering_within_cluster(
                ordered_clusters[cluster_id], true_indices
            )
            cluster_type = true_indices[ordered_clusters[cluster_id][0]][0].upper()
            print(f"  Cluster {cluster_id} ({cluster_type}):")
            print(f"    - Consecutive pairs: {metrics['consecutive_pairs']*100:.1f}%")
            print(f"    - LIS ratio: {metrics['lis_ratio']*100:.1f}%")
            print(f"    - Inversions: {metrics['inversions']}")
            print(f"    - Overall correlation: {metrics['spearman_corr']*100:.1f}%")
    
    # ===== SUMMARY =====
    print("\n" + "="*100)
    print("FORENSIC PIPELINE - COMPLETE")
    print("="*100)
    
    print(f"""
Pipeline Steps Completed:
  [✓] LOAD:        {n_fragments} fragments
  [✓] SHUFFLE:     Unknown order
  [✓] FEATURES:    256-dim byte frequency + 128-dim N-grams
  [✓] SIMILARITY:  Cosine distance
  [✓] CLUSTERING:  DBSCAN (eps={eps}) found {n_clusters} clusters
  [✓] ORDERING:    Greedy within clusters
  [✓] EVALUATION:  Quality metrics computed

Final Results:
  Clusters: {n_clusters}
  Total fragments: {sum(len(v) for v in ordered_clusters.values())}
  Algorithm: DBSCAN unsupervised (NO ground truth in pipeline)
""")
    print("="*100)


if __name__ == "__main__":
    main(file_types=["html", "pdf","apk"], frags_per_type=10)
