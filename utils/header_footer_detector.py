"""
Utility for detecting and removing headers and footers from file fragments
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FILE_SIGNATURES


def detect_header(file_bytes, file_type=None):
    """
    Detect if the file has a known header signature.
    
    Args:
        file_bytes: bytes object containing the file data
        file_type: optional file type hint (e.g., 'pdf', 'jpg')
    
    Returns:
        tuple: (has_header: bool, header_end_position: int, detected_type: str)
    """
    if file_type and file_type.lower() in FILE_SIGNATURES:
        # Check specific file type
        headers = FILE_SIGNATURES[file_type.lower()].get('header', [])
        for header_sig in headers:
            if file_bytes.startswith(header_sig):
                return True, len(header_sig), file_type.lower()
    else:
        # Check all known signatures
        for ftype, signatures in FILE_SIGNATURES.items():
            headers = signatures.get('header', [])
            for header_sig in headers:
                if file_bytes.startswith(header_sig):
                    return True, len(header_sig), ftype
    
    return False, 0, None


def detect_footer(file_bytes, file_type=None):
    """
    Detect if the file has a known footer signature.
    
    Args:
        file_bytes: bytes object containing the file data
        file_type: optional file type hint (e.g., 'pdf', 'jpg')
    
    Returns:
        tuple: (has_footer: bool, footer_start_position: int, detected_type: str)
    """
    if file_type and file_type.lower() in FILE_SIGNATURES:
        # Check specific file type
        footers = FILE_SIGNATURES[file_type.lower()].get('footer', [])
        for footer_sig in footers:
            if file_bytes.endswith(footer_sig):
                return True, len(file_bytes) - len(footer_sig), file_type.lower()
    else:
        # Check all known signatures
        for ftype, signatures in FILE_SIGNATURES.items():
            footers = signatures.get('footer', [])
            for footer_sig in footers:
                if file_bytes.endswith(footer_sig):
                    return True, len(file_bytes) - len(footer_sig), ftype
    
    return False, len(file_bytes), None


def remove_header(file_bytes, file_type=None, force_remove_bytes=None):
    """
    Remove header from file bytes.
    
    Args:
        file_bytes: bytes object containing the file data
        file_type: optional file type hint
        force_remove_bytes: if provided, remove this many bytes from start regardless of signature
    
    Returns:
        tuple: (cleaned_bytes: bytes, removed_bytes: int, had_header: bool)
    """
    if force_remove_bytes is not None:
        removed = min(force_remove_bytes, len(file_bytes))
        return file_bytes[removed:], removed, True
    
    has_header, header_end, detected_type = detect_header(file_bytes, file_type)
    
    if has_header:
        return file_bytes[header_end:], header_end, True
    
    return file_bytes, 0, False


def remove_footer(file_bytes, file_type=None, force_remove_bytes=None):
    """
    Remove footer from file bytes.
    
    Args:
        file_bytes: bytes object containing the file data
        file_type: optional file type hint
        force_remove_bytes: if provided, remove this many bytes from end regardless of signature
    
    Returns:
        tuple: (cleaned_bytes: bytes, removed_bytes: int, had_footer: bool)
    """
    if force_remove_bytes is not None:
        removed = min(force_remove_bytes, len(file_bytes))
        return file_bytes[:-removed] if removed > 0 else file_bytes, removed, True
    
    has_footer, footer_start, detected_type = detect_footer(file_bytes, file_type)
    
    if has_footer:
        removed_bytes = len(file_bytes) - footer_start
        return file_bytes[:footer_start], removed_bytes, True
    
    return file_bytes, 0, False


def remove_header_footer(file_bytes, file_type=None, force_header_bytes=None, force_footer_bytes=None):
    """
    Remove both header and footer from file bytes.
    
    Args:
        file_bytes: bytes object containing the file data
        file_type: optional file type hint
        force_header_bytes: if provided, remove this many bytes from start
        force_footer_bytes: if provided, remove this many bytes from end
    
    Returns:
        dict: {
            'cleaned_bytes': bytes,
            'original_size': int,
            'cleaned_size': int,
            'header_removed': int,
            'footer_removed': int,
            'had_header': bool,
            'had_footer': bool,
            'detected_type': str or None
        }
    """
    original_size = len(file_bytes)
    
    # Remove header
    cleaned_bytes, header_removed, had_header = remove_header(
        file_bytes, file_type, force_header_bytes
    )
    
    # Detect type from header if not provided
    detected_type = None
    if had_header and file_type is None:
        _, _, detected_type = detect_header(file_bytes)
    else:
        detected_type = file_type
    
    # Remove footer
    cleaned_bytes, footer_removed, had_footer = remove_footer(
        cleaned_bytes, detected_type, force_footer_bytes
    )
    
    cleaned_size = len(cleaned_bytes)
    
    return {
        'cleaned_bytes': cleaned_bytes,
        'original_size': original_size,
        'cleaned_size': cleaned_size,
        'header_removed': header_removed,
        'footer_removed': footer_removed,
        'had_header': had_header,
        'had_footer': had_footer,
        'detected_type': detected_type
    }


def analyze_file(file_path):
    """
    Analyze a file for header/footer presence.
    
    Args:
        file_path: path to the file
    
    Returns:
        dict: analysis results
    """
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    
    file_ext = os.path.splitext(file_path)[1].lstrip('.').lower()
    
    has_header, header_end, header_type = detect_header(file_bytes, file_ext)
    has_footer, footer_start, footer_type = detect_footer(file_bytes, file_ext)
    
    return {
        'file_path': file_path,
        'file_size': len(file_bytes),
        'file_extension': file_ext,
        'has_header': has_header,
        'header_end_position': header_end,
        'detected_type_from_header': header_type,
        'has_footer': has_footer,
        'footer_start_position': footer_start,
        'detected_type_from_footer': footer_type
    }


if __name__ == "__main__":
    # Test with a sample file
    import sys
    
    if len(sys.argv) > 1:
        result = analyze_file(sys.argv[1])
        print("\n=== File Analysis ===")
        print(f"File: {result['file_path']}")
        print(f"Size: {result['file_size']} bytes")
        print(f"Extension: {result['file_extension']}")
        print(f"\nHeader Detection:")
        print(f"  Has Header: {result['has_header']}")
        if result['has_header']:
            print(f"  Header Ends At: {result['header_end_position']}")
            print(f"  Detected Type: {result['detected_type_from_header']}")
        print(f"\nFooter Detection:")
        print(f"  Has Footer: {result['has_footer']}")
        if result['has_footer']:
            print(f"  Footer Starts At: {result['footer_start_position']}")
            print(f"  Detected Type: {result['detected_type_from_footer']}")
    else:
        print("Usage: python header_footer_detector.py <file_path>")
