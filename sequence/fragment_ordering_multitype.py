"""
Multi-Type Fragment Ordering Test
===================================
Professor's requirement:
1. Mix fragments from DIFFERENT file types (case_id 0001)
2. Shuffle them together
3. Try to find correct order
4. Compare with ground truth
5. Test scalability: 10-10 or 100-100 fragments per type

This tests if algorithm can order heterogeneous fragments (different file types mixed).
"""

import os
import re
import csv
import argparse
import numpy as np
import random
from collections import defaultdict
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fragment_ordering_optimized import (
    parse_fragment_name,
    get_files_for_type,
    compute_similarity_matrix,
    adaptive_order,
    evaluate_order,
    discover_file_types,
    fast_boundary_similarity
)


def load_fragments_by_type(dataset_root, file_types, case_id="0001", frags_per_type=10):
    """
    Load fragments from multiple file types for same case_id.
    
    Returns:
        mixed_dict: {filename: bytes} - all fragments mixed
        true_indices: {filename: original_index}
        type_mapping: {filename: file_type}
    """
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
            print(f"  Warning: Case {case_id} not found for {file_type}")
            continue
        
        frag_list = groups[case_id]
        frag_list.sort(key=lambda x: x[0])
        
        # Limit to frags_per_type
        frag_list = frag_list[:frags_per_type]
        
        # Load fragments
        for idx, fpath in frag_list:
            name = os.path.basename(fpath)
            with open(fpath, "rb") as f:
                mixed_dict[name] = f.read()
            
            # Store ground truth: (file_type, original_index)
            true_indices[name] = (file_type, idx)
            type_mapping[name] = file_type
    
    return mixed_dict, true_indices, type_mapping


def display_mixed_comparison(prediction, true_indices, type_mapping):
    """Display detailed comparison for mixed fragments."""
    print(f"\n  DETAILED RESULTS:")
    print(f"  {'-'*80}")
    print(f"  {'Filename':<40} {'FileType':<12} {'Index':<8} {'Status'}")
    print(f"  {'-'*80}")
    
    for i, fname in enumerate(prediction):
        file_type, idx = true_indices[fname]
        
        # Check if consecutive index is correct
        correct = "✓" if i == 0 else ""
        if i > 0:
            prev_fname = prediction[i-1]
            prev_type, prev_idx = true_indices[prev_fname]
            # Correct if: same type AND consecutive, OR transitions are valid
            if file_type == prev_type and idx == prev_idx + 1:
                correct = "✓ CORRECT"
            elif i > 0:
                correct = "✗ JUMP"
        
        print(f"  {fname:<40} {file_type:<12} {idx:<8} {correct}")


def run_multi_type_test(dataset_root, file_types, case_id="0001", frags_per_type=10, num_tests=1):
    """
    Test ordering with mixed fragments from different file types.
    """
    print(f"\n{'='*90}")
    print(f"  MULTI-TYPE FRAGMENT ORDERING TEST")
    print(f"{'='*90}")
    print(f"  File Types  : {', '.join(file_types)}")
    print(f"  Case ID     : {case_id}")
    print(f"  Frags/Type  : {frags_per_type}")
    print(f"  Total Frags : {len(file_types) * frags_per_type} (expected)")
    
    # Load fragments from all types
    mixed_dict, true_indices, type_mapping = load_fragments_by_type(
        dataset_root, file_types, case_id, frags_per_type
    )
    
    if not mixed_dict:
        print("  ERROR: No fragments loaded!")
        return None
    
    num_frags = len(mixed_dict)
    print(f"  Loaded     : {num_frags} fragments")
    print(f"\n  Fragments per type:")
    type_counts = defaultdict(int)
    for fname, (ftype, idx) in true_indices.items():
        type_counts[ftype] += 1
    for ftype in sorted(type_counts.keys()):
        print(f"    {ftype:<15}: {type_counts[ftype]}")
    
    # Show true order
    print(f"\n  STEP 1: TRUE (original) fragment order by file type")
    print(f"  {'-'*80}")
    for file_type in file_types:
        indices = sorted([idx for ftype, idx in true_indices.values() if ftype == file_type])
        if indices:
            print(f"    {file_type:<15}: {indices}")
    
    # Shuffle
    print(f"\n  STEP 2: SHUFFLING all {num_frags} fragments together...")
    shuffled_names = list(mixed_dict.keys())
    random.shuffle(shuffled_names)
    print(f"    Mixed! Fragments now in random order.")
    
    # Try to reorder
    print(f"\n  STEP 3: ATTEMPTING TO REORDER using fragment ordering algorithm...")
    sim_matrix, order_names, _ = compute_similarity_matrix(mixed_dict)
    predicted_order = adaptive_order(sim_matrix, order_names)
    
    # Evaluate
    print(f"\n  STEP 4: EVALUATION")
    
    # Count correct pairs within each file type
    within_type_correct = 0
    within_type_total = 0
    cross_type_transitions = 0
    
    for i in range(len(predicted_order) - 1):
        curr_name = predicted_order[i]
        next_name = predicted_order[i + 1]
        
        curr_type, curr_idx = true_indices[curr_name]
        next_type, next_idx = true_indices[next_name]
        
        if curr_type == next_type:
            within_type_total += 1
            if next_idx == curr_idx + 1:
                within_type_correct += 1
        else:
            cross_type_transitions += 1
    
    within_type_acc = within_type_correct / max(within_type_total, 1)
    print(f"  Within-Type Pair Accuracy : {within_type_acc:.2%} ({within_type_correct}/{within_type_total})")
    print(f"  Cross-Type Transitions    : {cross_type_transitions}")
    
    # Show detailed results
    display_mixed_comparison(predicted_order, true_indices, type_mapping)
    
    # Summary by type
    print(f"\n  STEP 5: PERFORMANCE BY FILE TYPE")
    print(f"  {'-'*80}")
    for file_type in file_types:
        frags_of_type = [name for name in predicted_order if type_mapping[name] == file_type]
        if frags_of_type:
            indices = [true_indices[name][1] for name in frags_of_type]
            is_sorted = indices == sorted(indices)
            status = "Correctly ordered" if is_sorted else "Out of order"
            print(f"  {file_type:<15}: {status}")
    
    print(f"\n  TIME TAKEN (estimate): Depends on fragment count and machine")
    print(f"  STATUS: {'GOOD - fragments grouping recognizable' if within_type_acc > 0.5 else 'POOR - mixing successful challenge'}")
    
    return {
        "NumFragments": num_frags,
        "WithinTypeAccuracy": round(within_type_acc, 4),
        "CrossTypeTransitions": cross_type_transitions,
        "FileTypes": len(file_types),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Type Fragment Ordering Test (Professor's Requirement)"
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "Train"),
        help="Path to Train folder"
    )
    parser.add_argument(
        "--file-types",
        default="html,json,css",
        help="Comma-separated file types (default: html,json,css)"
    )
    parser.add_argument(
        "--case-id",
        default="0001",
        help="Case ID to test (default: 0001)"
    )
    parser.add_argument(
        "--frags-per-type",
        type=int,
        default=10,
        help="Number of fragments per type (default: 10, try 100 if fast)"
    )
    parser.add_argument(
        "--output",
        default="multitype_results.csv",
        help="Output CSV file"
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        return
    
    file_types = [ft.strip().lower() for ft in args.file_types.split(",")]
    
    # Run test
    result = run_multi_type_test(
        args.dataset,
        file_types,
        case_id=args.case_id,
        frags_per_type=args.frags_per_type
    )
    
    # Save to CSV
    if result:
        file_exists = os.path.exists(args.output)
        mode = 'a' if file_exists else 'w'
        
        with open(args.output, mode, newline='', encoding='utf-8') as f:
            fieldnames = list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)
        
        action = "APPENDED to" if file_exists else "Saved to"
        print(f"\nResults {action}: {args.output}")


if __name__ == "__main__":
    main()
