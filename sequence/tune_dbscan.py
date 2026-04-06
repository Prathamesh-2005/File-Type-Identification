"""
DBSCAN PARAMETER TUNING
Find optimal eps and min_samples for forensic clustering
"""

import os
import sys
import numpy as np
from collections import defaultdict
import random

sys.path.insert(0, os.path.dirname(__file__))
from forensic_pipeline import (
    load_fragments,
    compute_similarity_matrix,
    cluster_fragments_dbscan,
    group_by_clusters,
)

def tune_dbscan(similarity_matrix, fragment_names, eps_range=None, min_samples_range=None):
    """
    Try multiple DBSCAN parameter combinations
    
    Args:
        similarity_matrix: cosine similarity matrix
        fragment_names: list of fragment names
        eps_range: list of eps values to try
        min_samples_range: list of min_samples to try
    
    Returns:
        results: list of {eps, min_samples, n_clusters, cluster_distribution}
    """
    if eps_range is None:
        eps_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    
    if min_samples_range is None:
        min_samples_range = [2, 3]
    
    results = []
    
    print("\n" + "="*100)
    print("DBSCAN PARAMETER TUNING")
    print("="*100)
    
    print(f"\nSimilarity matrix statistics:")
    print(f"  Min similarity: {similarity_matrix.min():.4f}")
    print(f"  Max similarity: {similarity_matrix.max():.4f}")
    print(f"  Mean similarity: {similarity_matrix.mean():.4f}")
    
    distance_matrix = 1.0 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    print(f"\nDistance matrix statistics:")
    print(f"  Min distance: {distance_matrix.min():.4f}")
    print(f"  Max distance: {distance_matrix.max():.4f}")
    print(f"  Mean distance: {distance_matrix.mean():.4f}")
    
    print(f"\n{'EPS':<8} {'MIN_SAMP':<10} {'N_CLUSTERS':<12} {'NOISE':<8} DISTRIBUTION")
    print("-" * 100)
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            labels, n_clusters = cluster_fragments_dbscan(
                similarity_matrix, fragment_names, eps=eps, min_samples=min_samples, verbose=False
            )
            
            # Don't print the detailed output from cluster_fragments_dbscan
            cluster_counts = defaultdict(int)
            for label in labels:
                cluster_counts[label] += 1
            
            n_noise = cluster_counts.get(-1, 0)
            
            # Format distribution string
            dist_str = ""
            for cid in sorted(cluster_counts.keys()):
                if cid == -1:
                    dist_str += f"Noise:{cluster_counts[cid]} "
                else:
                    dist_str += f"C{cid}:{cluster_counts[cid]} "
            
            print(f"{eps:<8.2f} {min_samples:<10} {n_clusters:<12} {n_noise:<8} {dist_str}")
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'labels': labels,
                'distribution': dist_str
            })
    
    return results


def main(dataset_root="Train", file_types=None, case_id="0001", frags_per_type=10):
    """Run DBSCAN parameter tuning"""
    if file_types is None:
        file_types = ["apk", "pdf"]
    
    # Load and prepare fragments
    fragments_dict, true_indices, file_list = load_fragments(
        dataset_root, file_types, case_id, frags_per_type
    )
    
    fragment_names = list(fragments_dict.keys())
    random.shuffle(fragment_names)
    
    # Compute features and similarity
    similarity_matrix, features = compute_similarity_matrix(fragments_dict, fragment_names)
    
    # Tune parameters
    results = tune_dbscan(
        similarity_matrix, fragment_names,
        eps_range=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        min_samples_range=[2, 3]
    )
    
    # Recommend best parameters
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)
    
    print("\n[OPTION 1] For separating into distinct clusters (APK vs PDF):")
    print("  Use: eps=0.2, min_samples=2")
    print("  Reason: Creates 2-3 clusters, separates file types")
    
    print("\n[OPTION 2] For grouping similar fragments within same file:")
    print("  Use: eps=0.15, min_samples=2")
    print("  Reason: Creates more granular clusters")
    
    print("\nTip: Lower eps → More clusters (stricter similarity threshold)")
    print("     Higher eps → Fewer clusters (looser similarity threshold)")


if __name__ == "__main__":
    main()
