"""
Script to generate fragments from complete files with optional header/footer removal
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RAW_DATA_DIR, FRAGMENTS_DIR, FRAGMENT_SIZE,
    DEFAULT_HEADER_SIZE, DEFAULT_FOOTER_SIZE
)
from utils.header_footer_detector import remove_header_footer, analyze_file


def generate_fragments(file_path, fragment_size=FRAGMENT_SIZE, 
                       remove_headers=False, remove_footers=False,
                       force_header_bytes=None, force_footer_bytes=None,
                       min_fragment_size=512):
    """
    Generate fragments from a complete file.
    
    Args:
        file_path: path to the input file
        fragment_size: size of each fragment in bytes
        remove_headers: boolean, whether to remove headers
        remove_footers: boolean, whether to remove footers
        force_header_bytes: if provided, remove this many bytes from start
        force_footer_bytes: if provided, remove this many bytes from end
        min_fragment_size: minimum size for a fragment to be valid
    
    Returns:
        list: list of fragment byte arrays
    """
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    # Get file type from extension
    file_ext = os.path.splitext(file_path)[1].lstrip('.').lower()
    
    # Remove header/footer if requested
    if remove_headers or remove_footers:
        result = remove_header_footer(
            file_bytes,
            file_type=file_ext,
            force_header_bytes=force_header_bytes if remove_headers else None,
            force_footer_bytes=force_footer_bytes if remove_footers else None
        )
        file_bytes = result['cleaned_bytes']
        print(f"  Original size: {result['original_size']} bytes")
        print(f"  Cleaned size: {result['cleaned_size']} bytes")
        print(f"  Header removed: {result['header_removed']} bytes (had_header: {result['had_header']})")
        print(f"  Footer removed: {result['footer_removed']} bytes (had_footer: {result['had_footer']})")
    
    # Generate fragments
    fragments = []
    num_fragments = len(file_bytes) // fragment_size
    
    for i in range(num_fragments):
        start = i * fragment_size
        end = start + fragment_size
        fragment = file_bytes[start:end]
        
        if len(fragment) >= min_fragment_size:
            fragments.append(fragment)
    
    # Handle remaining bytes
    if len(file_bytes) % fragment_size >= min_fragment_size:
        fragments.append(file_bytes[num_fragments * fragment_size:])
    
    return fragments


def process_directory(input_dir, output_dir, fragment_size=FRAGMENT_SIZE,
                      remove_headers=False, remove_footers=False,
                      force_header_bytes=None, force_footer_bytes=None,
                      supported_extensions=None):
    """
    Process all files in a directory and generate fragments.
    
    Args:
        input_dir: directory containing original files
        output_dir: directory to save fragments
        fragment_size: size of each fragment
        remove_headers: whether to remove headers
        remove_footers: whether to remove footers
        force_header_bytes: forced header removal size
        force_footer_bytes: forced footer removal size
        supported_extensions: list of file extensions to process
    
    Returns:
        DataFrame: mapping of fragments to original files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fragment_mapping = []
    fragment_id = 0
    
    # Walk through input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lstrip('.').lower()
            
            # Skip if extension not supported
            if supported_extensions and file_ext not in supported_extensions:
                continue
            
            print(f"\nProcessing: {filename}")
            
            try:
                # Generate fragments
                fragments = generate_fragments(
                    file_path,
                    fragment_size=fragment_size,
                    remove_headers=remove_headers,
                    remove_footers=remove_footers,
                    force_header_bytes=force_header_bytes,
                    force_footer_bytes=force_footer_bytes
                )
                
                # Create output subdirectory for this file type
                type_output_dir = os.path.join(output_dir, file_ext)
                os.makedirs(type_output_dir, exist_ok=True)
                
                # Save fragments
                for idx, fragment in enumerate(fragments):
                    fragment_filename = f"{Path(filename).stem}_frag_{idx}.npy"
                    fragment_path = os.path.join(type_output_dir, fragment_filename)
                    
                    # Convert to numpy array and save
                    fragment_array = np.frombuffer(fragment, dtype=np.uint8)
                    np.save(fragment_path, fragment_array)
                    
                    # Record mapping
                    fragment_mapping.append({
                        'fragment_id': fragment_id,
                        'fragment_filename': fragment_filename,
                        'fragment_path': fragment_path,
                        'original_file': filename,
                        'original_path': file_path,
                        'file_type': file_ext,
                        'fragment_index': idx,
                        'fragment_size': len(fragment)
                    })
                    
                    fragment_id += 1
                
                print(f"  Generated {len(fragments)} fragments")
                
            except Exception as e:
                print(f"  Error processing {filename}: {str(e)}")
                continue
    
    # Create DataFrame and save mapping
    df = pd.DataFrame(fragment_mapping)
    mapping_path = os.path.join(output_dir, 'fragment_mapping.csv')
    df.to_csv(mapping_path, index=False)
    print(f"\n✓ Fragment mapping saved to: {mapping_path}")
    print(f"✓ Total fragments generated: {len(df)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate file fragments with optional header/footer removal'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default=RAW_DATA_DIR,
        help='Input directory containing original files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=FRAGMENTS_DIR,
        help='Output directory for fragments'
    )
    
    parser.add_argument(
        '--fragment-size',
        type=int,
        default=FRAGMENT_SIZE,
        help='Size of each fragment in bytes'
    )
    
    parser.add_argument(
        '--remove-headers',
        action='store_true',
        help='Remove file headers (use signature detection)'
    )
    
    parser.add_argument(
        '--remove-footers',
        action='store_true',
        help='Remove file footers (use signature detection)'
    )
    
    parser.add_argument(
        '--force-header-bytes',
        type=int,
        default=None,
        help='Force removal of N bytes from start (overrides signature detection)'
    )
    
    parser.add_argument(
        '--force-footer-bytes',
        type=int,
        default=None,
        help='Force removal of N bytes from end (overrides signature detection)'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=None,
        help='List of file extensions to process (e.g., pdf jpg mp3)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FILE FRAGMENT GENERATION")
    print("=" * 60)
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Fragment Size: {args.fragment_size} bytes")
    print(f"Remove Headers: {args.remove_headers}")
    print(f"Remove Footers: {args.remove_footers}")
    
    if args.force_header_bytes:
        print(f"Force Header Removal: {args.force_header_bytes} bytes")
    if args.force_footer_bytes:
        print(f"Force Footer Removal: {args.force_footer_bytes} bytes")
    if args.extensions:
        print(f"Extensions: {', '.join(args.extensions)}")
    
    print("=" * 60)
    
    # Process directory
    df = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fragment_size=args.fragment_size,
        remove_headers=args.remove_headers,
        remove_footers=args.remove_footers,
        force_header_bytes=args.force_header_bytes,
        force_footer_bytes=args.force_footer_bytes,
        supported_extensions=args.extensions
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(df['file_type'].value_counts())
    print("\n✓ Fragment generation complete!")


if __name__ == "__main__":
    main()
