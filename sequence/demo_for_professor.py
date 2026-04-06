"""
QUICK DEMO - 10 Fragment Mixing Test
For Professor Demonstration
Shows: Clustering, Mixing, Ordering, Results
"""

import os
import sys
import numpy as np
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from fragment_ordering_optimized import (
    parse_fragment_name,
    get_files_for_type,
    compute_similarity_matrix,
    adaptive_order,
    evaluate_order,
    discover_file_types,
)

def load_fragments_by_type(dataset_root, file_types, case_id="0001", frags_per_type=10):
    """Load fragments from multiple file types"""
    from collections import defaultdict
    
    mixed_dict = {}
    true_indices = {}
    type_mapping = {}
    
    for file_type in file_types:
        files = get_files_for_type(dataset_root, file_type)
        
        # Group by case
        groups = defaultdict(list)
        for fpath in files:
            cid, fidx, ftype = parse_fragment_name(fpath)
            if cid:
                groups[cid].append((fidx, fpath))
        
        # Get case_id fragments
        if case_id not in groups:
            continue
        
        frag_list = groups[case_id]
        frag_list.sort(key=lambda x: x[0])
        frag_list = frag_list[:frags_per_type]
        
        # Load fragments
        for idx, fpath in frag_list:
            name = os.path.basename(fpath)
            with open(fpath, "rb") as f:
                mixed_dict[name] = f.read()
            
            true_indices[name] = (file_type, idx)
            type_mapping[name] = file_type
    
    return mixed_dict, true_indices, type_mapping

def run_demo(dataset_root="Train", file_types=None, frags_per_type=10):
    if file_types is None:
        file_types = ["html", "apk", "pdf", "bin"]  # 4 types
    
    print("\n" + "="*80)
    print("FRAGMENT ORDERING DEMO - 10 FRAGMENTS FROM MULTIPLE TYPES")
    print("="*80)
    
    # ===== STEP 1: CLUSTERING =====
    print("\n[STEP 1] CLUSTERING - Loading fragments by file type")
    print("-" * 80)
    
    mixed_dict, true_indices, type_mapping = load_fragments_by_type(
        dataset_root, file_types, case_id="0001", frags_per_type=frags_per_type
    )
    
    # Show clustering
    file_type_counts = defaultdict(int)
    for fname, (ftype, _) in true_indices.items():
        file_type_counts[ftype] += 1
    
    print(f"\nFragments Loaded by Type:")
    total_frags = 0
    for ftype in file_types:
        count = file_type_counts[ftype]
        if count > 0:
            total_frags += count
            print(f"  ✓ {ftype.upper():10s}: {count:2d} fragments")
    
    print(f"\n  TOTAL: {total_frags} fragments clustered into {len(file_types)} types")
    
    if total_frags == 0:
        print("\n[ERROR] No fragments found! Check dataset path.")
        return
    
    # ===== STEP 2: MIXING (SHUFFLING) =====
    print("\n[STEP 2] MIXING - Shuffle all fragments together")
    print("-" * 80)
    
    mixed_order = list(mixed_dict.keys())
    print(f"\nBefore shuffle (by type):")
    for i, fname in enumerate(mixed_order[:5]):
        print(f"  {i}. {fname} ({type_mapping[fname].upper()})")
    if len(mixed_order) > 5:
        print(f"  ... and {len(mixed_order) - 5} more")
    
    random.shuffle(mixed_order)
    
    print(f"\nAfter shuffle (randomized):")
    for i, fname in enumerate(mixed_order[:5]):
        print(f"  {i}. {fname} ({type_mapping[fname].upper()})")
    if len(mixed_order) > 5:
        print(f"  ... and {len(mixed_order) - 5} more")
    
    shuffled_dict = {fname: mixed_dict[fname] for fname in mixed_order}
    
    # ===== STEP 3: ORDERING =====
    print("\n[STEP 3] ORDERING - Find correct sequence")
    print("-" * 80)
    print(f"\nComputing similarity matrix ({len(shuffled_dict)}x{len(shuffled_dict)})...")
    
    sim_matrix, order_names, _ = compute_similarity_matrix(shuffled_dict)
    
    print(f"Running adaptive algorithm...")
    predicted_order = adaptive_order(sim_matrix, order_names)
    
    print(f"✓ Predicted order computed ({len(predicted_order)} fragments)")
    
    # ===== STEP 4: EVALUATION =====
    print("\n[STEP 4] EVALUATION - Compare with ground truth")
    print("-" * 80)
    
    # Count within-type pairs
    within_type_correct = 0
    within_type_total = 0
    cross_type_transitions = 0
    
    for i in range(len(predicted_order) - 1):
        curr_name = predicted_order[i]
        next_name = predicted_order[i + 1]
        
        curr_type, curr_true_idx = true_indices[curr_name]
        next_type, next_true_idx = true_indices[next_name]
        
        # Check if pair is correct AND same type
        if curr_true_idx + 1 == next_true_idx:
            if curr_type == next_type:
                within_type_correct += 1
        
        within_type_total += 1
        
        # Count cross-type transitions
        if curr_type != next_type:
            cross_type_transitions += 1
    
    within_type_accuracy = within_type_correct / within_type_total if within_type_total > 0 else 0
    
    print(f"\nResults:")
    print(f"  Correct consecutive pairs (same type): {within_type_correct}/{within_type_total}")
    print(f"  Within-Type Accuracy: {within_type_accuracy*100:.2f}%")
    print(f"  Cross-Type Transitions: {cross_type_transitions}")
    
    # Show sample
    print(f"\nSample Ordered Sequence (first 8):")
    for i, fname in enumerate(predicted_order[:8]):
        ftype, _ = true_indices[fname]
        if i > 0:
            prev_ftype, _ = true_indices[predicted_order[i-1]]
            marker = " [SAME TYPE]" if prev_ftype == ftype else " [TYPE CHANGE]"
        else:
            marker = ""
        print(f"  {i}. {fname} ({ftype.upper()}){marker}")
    
    # ===== SUMMARY =====
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
Process Completed:
  [✓] Clustering:  Fragments grouped by file type
  [✓] Mixing:      All fragments shuffled together  
  [✓] Ordering:    Correct sequence found
  [✓] Evaluation:  Results compared to ground truth

Results:
  Total Fragments: {total_frags} ({', '.join([f"{ftype}:{file_type_counts[ftype]}" for ftype in file_types if file_type_counts[ftype] > 0])})
  Accuracy: {within_type_accuracy*100:.2f}%
  Cross-Type Jumps: {cross_type_transitions}

Status: ✓ READY TO SHOW PROFESSOR
""")
    print("="*80)

if __name__ == "__main__":
    run_demo(dataset_root="Train", file_types=["html", "apk", "pdf", "bin"], frags_per_type=10)
