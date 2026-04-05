"""
Step 2: Same-File Fragment Ordering
====================================
For each file type folder (e.g., apkFragments), and for each original file
(e.g., 0001), predict the correct order of its fragments using byte content
only (no filename peeking), then compare against the true order.

Dataset naming convention:
    0001-apk_trimmed_frag0_apk.bin
    0001-apk_trimmed_frag1_apk.bin
    ...

Usage examples:
    python step2_forensic_final.py                        # all file types
    python step2_forensic_final.py --file-type apk        # one type only
    python step2_forensic_final.py --dataset C:/Train     # custom dataset path
    python step2_forensic_final.py --no-plot              # skip charts
"""

import os
import re
import csv
import zlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from itertools import permutations
from sklearn.metrics import adjusted_rand_score

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train")
BOUNDARY  = 128   # bytes used for boundary similarity check
TOP_N_NGRAMS = 300  # top N n-gram features to keep (for speed)

# ─────────────────────────────────────────────
# FILENAME PARSING
# ─────────────────────────────────────────────

def parse_fragment_name(filename):
    """
    Parse '0001-apk_trimmed_frag2_apk.bin'
    Returns (case_id='0001', frag_index=2, file_type='apk')
    """
    name = os.path.basename(filename).lower()
    m = re.search(r'^([\w\-]+?)_frag(\d+)_([a-z0-9]+)\.bin$', name)
    if m:
        case_id   = m.group(1)         # e.g. '0001-apk_trimmed'
        frag_idx  = int(m.group(2))    # e.g. 2
        file_type = m.group(3)         # e.g. 'apk'
        # Normalize case_id to just the leading numeric part if possible
        id_match = re.match(r'^(\d+)', case_id)
        short_id = id_match.group(1) if id_match else case_id
        return short_id, frag_idx, file_type
    return None, None, None


def get_files_for_type(dataset_root, file_type):
    """
    Find all .bin fragments inside <file_type>Fragments folder.
    Returns list of full paths.
    """
    folder = os.path.join(dataset_root, f"{file_type}Fragments")
    if not os.path.isdir(folder):
        # Also try exact match
        folder = os.path.join(dataset_root, file_type)
    if not os.path.isdir(folder):
        print(f"  [WARN] Folder not found for type '{file_type}'")
        return []
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".bin")
    ]
    return files


def discover_file_types(dataset_root):
    """
    Discover all file types from folder names like 'apkFragments'.
    """
    types = []
    for name in os.listdir(dataset_root):
        if os.path.isdir(os.path.join(dataset_root, name)):
            lower = name.lower()
            if lower.endswith("fragments"):
                types.append(lower[:-9])   # strip 'fragments'
            else:
                types.append(lower)
    return sorted(types)


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def byte_frequency_vector(data):
    """256-dim normalized byte frequency vector."""
    arr = np.frombuffer(data, dtype=np.uint8)
    vec = np.bincount(arr, minlength=256).astype(float)
    total = vec.sum()
    return vec / total if total > 0 else vec


def ngram_counter(data, n=4):
    """Return Counter of byte n-grams."""
    return Counter(data[i:i+n] for i in range(len(data) - n + 1))


def build_ngram_feature(data, vocab, n=4):
    """Build a fixed-size n-gram frequency vector using a shared vocabulary."""
    counter = ngram_counter(data, n)
    vec = np.array([counter.get(gram, 0) for gram in vocab], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else vec


def ngram_similarity(a, b, n=4):
    """Jaccard similarity of byte n-gram sets."""
    if len(a) < n or len(b) < n:
        return 0.0

    grams_a = set(a[i:i+n] for i in range(len(a)-n+1))
    grams_b = set(b[i:i+n] for i in range(len(b)-n+1))

    if not grams_a or not grams_b:
        return 0.0

    inter = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return inter / union if union > 0 else 0.0


def build_vocabulary(all_data_list, n=4, top_k=TOP_N_NGRAMS):
    """
    Build shared n-gram vocabulary from all fragments.
    Returns list of top-k most common n-grams.
    """
    total_counter = Counter()
    for data in all_data_list:
        total_counter.update(ngram_counter(data, n))
    return [gram for gram, _ in total_counter.most_common(top_k)]


def extract_combined_features(fragments_data):
    """
    For a dict {filename: bytes}, return feature matrix and ordered name list.
    Features = byte_freq (256-dim) + ngram_freq (TOP_N_NGRAMS-dim)
    """
    names = list(fragments_data.keys())
    data_list = [fragments_data[n] for n in names]

    # Build shared vocabulary from these fragments
    vocab = build_vocabulary(data_list)

    features = []
    for data in data_list:
        bf = byte_frequency_vector(data)
        nf = build_ngram_feature(data, vocab)
        combined = np.concatenate([bf, nf])
        features.append(combined)

    return np.array(features), names


# ─────────────────────────────────────────────
# BOUNDARY SIMILARITY (for ordering)
# ─────────────────────────────────────────────

def boundary_similarity(tail_data, head_data):
    """
    Compare the last BOUNDARY bytes of fragment A
    with the first BOUNDARY bytes of fragment B.
    Returns a similarity score (higher = more likely consecutive).
    """
    tail = np.frombuffer(tail_data[-BOUNDARY:], dtype=np.uint8).astype(float)
    head = np.frombuffer(head_data[:BOUNDARY],  dtype=np.uint8).astype(float)

    # Byte match score
    byte_match = np.sum(tail == head) / BOUNDARY

    # Cosine similarity
    tail_norm = np.linalg.norm(tail)
    head_norm = np.linalg.norm(head)
    if tail_norm > 0 and head_norm > 0:
        cosine = np.dot(tail, head) / (tail_norm * head_norm)
    else:
        cosine = 0.0

    # N-gram continuity over boundary windows (direction-aware)
    ngram_score = ngram_similarity(tail_data[-BOUNDARY:], head_data[:BOUNDARY], n=4)

    # Compression gain when concatenating in this direction.
    # If A->B is more "natural", A+B tends to compress slightly better than random ordering.
    comp_a = len(zlib.compress(tail_data, level=1))
    comp_b = len(zlib.compress(head_data, level=1))
    comp_ab = len(zlib.compress(tail_data + head_data, level=1))
    comp_gain = (comp_a + comp_b - comp_ab) / max(comp_a + comp_b, 1)

    # Weighted transition score (content-only, no filename info)
    return (0.15 * byte_match) + (0.30 * cosine) + (0.20 * ngram_score) + (0.35 * comp_gain)


# ─────────────────────────────────────────────
# FRAGMENT ORDERING
# ─────────────────────────────────────────────

def order_fragments_greedy(fragment_data_dict):
    """
    Given {name: bytes}, predict the best ordering using greedy nearest-neighbor
    boundary similarity. Returns ordered list of names.
    """
    names = list(fragment_data_dict.keys())

    if len(names) == 1:
        return names

    if len(names) == 2:
        a, b = names[0], names[1]
        score_ab = boundary_similarity(fragment_data_dict[a], fragment_data_dict[b])
        score_ba = boundary_similarity(fragment_data_dict[b], fragment_data_dict[a])
        return [a, b] if score_ab >= score_ba else [b, a]

    # Try each fragment as the starting point, pick best total score
    best_order = None
    best_score = -1.0

    # For small sets (<=6), try a few starts; for larger, just try all
    starts = names if len(names) <= 8 else names[:5]

    for start in starts:
        remaining = set(names) - {start}
        order = [start]
        current = start
        total_score = 0.0

        while remaining:
            best_next = None
            best_next_score = -1.0
            for candidate in remaining:
                score = boundary_similarity(
                    fragment_data_dict[current],
                    fragment_data_dict[candidate]
                )
                if score > best_next_score:
                    best_next_score = score
                    best_next = candidate
            order.append(best_next)
            total_score += best_next_score
            remaining.remove(best_next)
            current = best_next

        if total_score > best_score:
            best_score = total_score
            best_order = order

    return best_order


def two_fragment_margin(fragment_data_dict):
    """Return directional scores and absolute margin for 2-fragment cases."""
    names = list(fragment_data_dict.keys())
    if len(names) != 2:
        return None, None, None

    a, b = names[0], names[1]
    score_ab = boundary_similarity(fragment_data_dict[a], fragment_data_dict[b])
    score_ba = boundary_similarity(fragment_data_dict[b], fragment_data_dict[a])
    return score_ab, score_ba, abs(score_ab - score_ba)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_ordering(predicted_order, true_indices):
    """
    predicted_order : list of filenames in predicted sequence
    true_indices    : dict {filename: true_frag_index}

    Returns:
        pair_accuracy  - fraction of consecutive pairs correctly ordered
        full_accuracy  - 1.0 if entire sequence is perfectly ordered
    """
    n = len(predicted_order)
    if n <= 1:
        return 1.0, 1.0

    correct_pairs = 0
    total_pairs = n - 1

    for i in range(total_pairs):
        a = predicted_order[i]
        b = predicted_order[i + 1]
        if true_indices[b] == true_indices[a] + 1:
            correct_pairs += 1

    pair_acc = correct_pairs / total_pairs

    # Full sequence correct?
    predicted_idx_seq = [true_indices[f] for f in predicted_order]
    expected_seq = sorted(predicted_idx_seq)
    full_acc = 1.0 if predicted_idx_seq == expected_seq else 0.0

    return pair_acc, full_acc


# ─────────────────────────────────────────────
# MAIN PIPELINE PER FILE TYPE
# ─────────────────────────────────────────────

def run_for_file_type(file_type, dataset_root, show_plots=True, case_id_filter="0001"):
    """
    Run the ordering pipeline for one file type.
    Returns list of result dicts for CSV export.
    """
    print(f"\n{'='*65}")
    print(f"  FILE TYPE: {file_type.upper()}")
    print(f"{'='*65}")

    files = get_files_for_type(dataset_root, file_type)
    if not files:
        print(f"  No fragments found. Skipping.")
        return []

    # Group files by case_id (original file)
    groups = defaultdict(list)
    for fpath in files:
        case_id, frag_idx, ftype = parse_fragment_name(fpath)
        if case_id is None:
            print(f"  [WARN] Could not parse: {os.path.basename(fpath)}")
            continue
        groups[case_id].append((frag_idx, fpath))

    print(f"  Found {len(groups)} original file(s), {len(files)} total fragments")

    # Restrict run to one original case (default: 0001)
    if case_id_filter:
        groups = {k: v for k, v in groups.items() if k == str(case_id_filter)}
        if not groups:
            print(f"  [INFO] Case '{case_id_filter}' not found for type '{file_type}'. Skipping.")
            return []
        print(f"  Running only case: {case_id_filter}")

    results = []
    all_pair_accs = []

    for case_id, frag_list in sorted(groups.items()):
        # Sort by true index for loading
        frag_list.sort(key=lambda x: x[0])
        true_order = {os.path.basename(fp): idx for idx, fp in frag_list}

        # Load fragment bytes
        fragment_data = {}
        for _, fpath in frag_list:
            name = os.path.basename(fpath)
            with open(fpath, "rb") as f:
                fragment_data[name] = f.read()

        num_frags = len(fragment_data)

        # Predict ordering using byte content only (no filename)
        predicted_order = order_fragments_greedy(fragment_data)

        # Evaluate
        pair_acc, full_acc = evaluate_ordering(predicted_order, true_order)
        all_pair_accs.append(pair_acc)

        # Build predicted vs actual string
        pred_indices = [true_order[f] for f in predicted_order]
        true_indices_sorted = sorted(true_order.values())
        status = "PERFECT" if full_acc == 1.0 else "PARTIAL" if pair_acc > 0 else "WRONG"

        # For 2-fragment files, flag low-confidence directional decisions.
        score_ab, score_ba, margin = two_fragment_margin(fragment_data)
        if margin is not None:
            print(f"  Direction scores: A->B={score_ab:.4f}, B->A={score_ba:.4f}, margin={margin:.4f}")
            if margin < 0.04:
                status = "AMBIGUOUS_LOW_SIGNAL"

        print(f"\n  Case: {case_id} | Fragments: {num_frags}")
        print(f"  True order   : {true_indices_sorted}")
        print(f"  Predicted    : {pred_indices}")
        print(f"  Pair Accuracy: {pair_acc:.2f}  |  Full Match: {'YES' if full_acc==1.0 else 'NO'}  |  {status}")

        results.append({
            "FileType":      file_type,
            "CaseID":        case_id,
            "NumFragments":  num_frags,
            "TrueOrder":     str(true_indices_sorted),
            "PredictedOrder":str(pred_indices),
            "PairAccuracy":  round(pair_acc, 4),
            "FullMatch":     "YES" if full_acc == 1.0 else "NO",
            "Status":        status,
        })

    # Summary
    avg_acc = np.mean(all_pair_accs) if all_pair_accs else 0.0
    perfect_count = sum(1 for r in results if r["FullMatch"] == "YES")
    print(f"\n  ── Summary for {file_type.upper()} ──")
    print(f"  Avg Pair Accuracy : {avg_acc:.4f}")
    print(f"  Perfect Orderings : {perfect_count}/{len(results)}")

    # Bar chart
    if show_plots and results:
        case_ids  = [r["CaseID"] for r in results]
        pair_accs = [r["PairAccuracy"] for r in results]
        colors = ["green" if a == 1.0 else "orange" if a > 0 else "red"
                  for a in pair_accs]
        plt.figure(figsize=(max(8, len(case_ids)), 4))
        bars = plt.bar(case_ids, pair_accs, color=colors)
        plt.axhline(y=avg_acc, color="blue", linestyle="--",
                    label=f"Avg = {avg_acc:.2f}")
        plt.title(f"Fragment Ordering Accuracy — {file_type.upper()}")
        plt.xlabel("Case ID")
        plt.ylabel("Pair Accuracy")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results_{file_type}.png", dpi=100)
        plt.show()
        print(f"  Chart saved: results_{file_type}.png")

    return results


# ─────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────

def save_results_csv(all_results, output_path="final_output_results.csv", append=True):
    if not all_results:
        print("No results to save.")
        return

    fieldnames = [
        "FileType", "CaseID", "NumFragments",
        "TrueOrder", "PredictedOrder",
        "PairAccuracy", "FullMatch", "Status"
    ]
    file_exists = os.path.exists(output_path)
    mode = "a" if append else "w"

    # If appending to an existing file without header, fix it once.
    if append and file_exists:
        with open(output_path, "r", encoding="utf-8") as check_f:
            first_line = check_f.readline().strip()

        expected_header = "FileType,CaseID,NumFragments,TrueOrder,PredictedOrder,PairAccuracy,FullMatch,Status"
        if first_line != expected_header:
            with open(output_path, "r", encoding="utf-8") as old_f:
                old_content = old_f.read()
            with open(output_path, "w", newline="", encoding="utf-8") as new_f:
                new_f.write(expected_header + "\n")
                new_f.write(old_content)

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if (not append) or (append and not file_exists):
            writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {output_path}")

    # Print final summary table
    print(f"\n{'='*65}")
    print("FINAL SUMMARY")
    print(f"{'='*65}")
    print(f"{'FileType':<15} {'Cases':<8} {'AvgPairAcc':<14} {'PerfectFull'}")
    print(f"{'-'*65}")

    from itertools import groupby
    key = lambda r: r["FileType"]
    for ftype, group in groupby(sorted(all_results, key=key), key=key):
        rows = list(group)
        avg  = np.mean([r["PairAccuracy"] for r in rows])
        perf = sum(1 for r in rows if r["FullMatch"] == "YES")
        print(f"{ftype:<15} {len(rows):<8} {avg:<14.4f} {perf}/{len(rows)}")

    print(f"{'='*65}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Fragment ordering pipeline"
    )
    parser.add_argument(
        "--dataset", default=DATASET,
        help=f"Path to Train folder (default: {DATASET})"
    )
    parser.add_argument(
        "--file-type", default=None,
        help="Run for one file type only, e.g. apk, bmp, pdf"
    )
    parser.add_argument(
        "--output", default="final_output_results.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--case-id", default="0001",
        help="Run only this original case id (default: 0001)"
    )
    parser.add_argument(
        "--all-cases", action="store_true",
        help="Process all case IDs (disables --case-id filter)"
    )
    parser.add_argument(
        "--reset-output", action="store_true",
        help="Delete existing output CSV before run"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable charts (faster)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"ERROR: Dataset folder not found: {args.dataset}")
        return

    show_plots = not args.no_plot

    # Keep one CSV output file across runs
    if args.reset_output and os.path.exists(args.output):
        os.remove(args.output)
        print(f"Reset output file: {args.output}")

    case_id_filter = None if args.all_cases else args.case_id

    if args.file_type:
        file_types = [args.file_type.lower()]
    else:
        file_types = discover_file_types(args.dataset)
        print(f"Discovered {len(file_types)} file type(s): {file_types}")

    all_results = []
    for ft in file_types:
        results = run_for_file_type(
            ft,
            args.dataset,
            show_plots=show_plots,
            case_id_filter=case_id_filter
        )
        all_results.extend(results)

    save_results_csv(all_results, args.output, append=True)


if __name__ == "__main__":
    main()