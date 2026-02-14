"""
Analyze fragments to check if they are headerless/footerless and count dataset statistics.
"""
import os
import csv
from pathlib import Path

# Known file signatures (magic numbers)
FILE_SIGNATURES = {
    'pdf': b'%PDF',
    'mp3': [b'\xff\xfb', b'\xff\xf3', b'\xff\xf2', b'ID3'],
    'mp4': [b'ftyp', b'moov'],
    '7zip': b'7z\xbc\xaf\x27\x1c',
    'apk': b'PK\x03\x04',  # ZIP signature
    'elf': b'\x7fELF',
    'html': [b'<!DOCTYPE', b'<html', b'<HTML'],
    'css': b'',  # No specific signature
    'javascript': b'',  # No specific signature
    'json': [b'{', b'['],
    'bin': b'',  # No specific signature
    'rtf': b'{\\rtf',
    'tif': [b'II\x2a\x00', b'MM\x00\x2a'],  # Little/Big endian TIFF
    'xlsx': b'PK\x03\x04',  # ZIP signature
}

def check_header(data, file_type):
    """Check if fragment contains file header."""
    if file_type not in FILE_SIGNATURES:
        return False
    
    sig = FILE_SIGNATURES[file_type]
    if isinstance(sig, list):
        return any(data.startswith(s) for s in sig)
    elif sig:
        return data.startswith(sig)
    return False

def analyze_fragment(filepath, file_type):
    """Analyze a single fragment."""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    has_header = check_header(data, file_type)
    return {
        'size': len(data),
        'has_header': has_header,
        'first_16_bytes': data[:16].hex(),
        'last_16_bytes': data[-16:].hex()
    }

def main():
    base_dir = Path(r'c:\Users\prath\Desktop\file-type-identification\Train')
    
    print("=" * 80)
    print("FRAGMENT ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    total_fragments = 0
    all_results = {}
    
    # Get all fragment directories
    fragment_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    for frag_dir in fragment_dirs:
        file_type = frag_dir.name.replace('Fragments', '').lower()
        
        # Find CSV file
        csv_files = list(frag_dir.glob('labels_*.csv'))
        if not csv_files:
            print(f"⚠️  No CSV file found in {frag_dir.name}")
            continue
        
        csv_file = csv_files[0]
        
        # Count fragments
        with open(csv_file, 'r') as f:
            fragment_count = len(f.readlines()) - 1  # Exclude header
        
        total_fragments += fragment_count
        
        # Analyze a sample fragment
        bin_files = list(frag_dir.glob('*.bin'))
        if bin_files:
            sample_file = bin_files[0]
            analysis = analyze_fragment(sample_file, file_type)
            
            print(f"📁 {frag_dir.name}")
            print(f"   Fragment Count: {fragment_count:,}")
            print(f"   Fragment Size:  {analysis['size']:,} bytes")
            print(f"   Has Header:     {analysis['has_header']}")
            print(f"   First 16 bytes: {analysis['first_16_bytes']}")
            print(f"   Last 16 bytes:  {analysis['last_16_bytes']}")
            print()
            
            all_results[file_type] = {
                'count': fragment_count,
                'size': analysis['size'],
                'has_header': analysis['has_header']
            }
    
    print("=" * 80)
    print(f"TOTAL FRAGMENTS: {total_fragments:,}")
    print("=" * 80)
    print()
    
    # Summary
    headerless_count = sum(1 for r in all_results.values() if not r['has_header'])
    print(f"✅ Headerless fragments: {headerless_count} out of {len(all_results)} file types")
    print()
    
    if headerless_count == len(all_results):
        print("✅ ALL FRAGMENTS ARE HEADERLESS - Ready for training!")
    else:
        print("⚠️  Some fragments contain headers:")
        for ft, res in all_results.items():
            if res['has_header']:
                print(f"   - {ft}")
    
    print()
    print("Fragment sizes:", set(r['size'] for r in all_results.values()))

if __name__ == '__main__':
    main()
