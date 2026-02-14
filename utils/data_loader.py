"""
Data loader for fragmented file type classification.
Loads fragments from CSV-labeled directories.
"""

import os
import csv
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import LabelEncoder
import pickle

class FragmentDataLoader:
    """Load and preprocess file fragments for classification."""
    
    def __init__(self, data_dir: str, fragment_size: int = 4096):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to directory containing fragment subdirectories
            fragment_size: Expected size of each fragment in bytes
        """
        self.data_dir = Path(data_dir)
        self.fragment_size = fragment_size
        self.label_encoder = LabelEncoder()
        self.file_types = []
        
    def get_file_types(self) -> List[str]:
        """Get list of available file types (from subdirectories)."""
        fragment_dirs = [d for d in self.data_dir.iterdir() 
                        if d.is_dir() and 'Fragments' in d.name]
        
        file_types = []
        for frag_dir in fragment_dirs:
            # Extract file type from directory name (e.g., "pdfFragments" -> "pdf")
            file_type = frag_dir.name.replace('Fragments', '').lower()
            
            # Check if CSV file exists
            csv_files = list(frag_dir.glob('labels_*.csv'))
            if csv_files:
                file_types.append(file_type)
        
        return sorted(file_types)
    
    def load_fragments(self, file_type: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load fragments for a specific file type.
        
        Args:
            file_type: Type of file (e.g., 'pdf', 'mp3')
            
        Returns:
            Tuple of (fragment_data, fragment_names)
        """
        frag_dir = self.data_dir / f"{file_type}Fragments"
        
        if not frag_dir.exists():
            raise ValueError(f"Fragment directory not found: {frag_dir}")
        
        # Find CSV label file
        csv_files = list(frag_dir.glob(f'labels_{file_type}.csv'))
        if not csv_files:
            raise ValueError(f"No labels CSV found in {frag_dir}")
        
        csv_file = csv_files[0]
        
        # Read fragment names from CSV
        fragments = []
        fragment_names = []
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fragment_name = row['fragment_name']
                fragment_path = frag_dir / fragment_name
                
                if fragment_path.exists():
                    try:
                        with open(fragment_path, 'rb') as frag_file:
                            data = frag_file.read()
                            
                        # Verify size
                        if len(data) == self.fragment_size:
                            fragments.append(np.frombuffer(data, dtype=np.uint8))
                            fragment_names.append(fragment_name)
                        else:
                            print(f"⚠️  Skipping {fragment_name}: size {len(data)} != {self.fragment_size}")
                            
                    except Exception as e:
                        print(f"⚠️  Error loading {fragment_name}: {e}")
        
        return np.array(fragments), fragment_names
    
    def load_all(self, normalize: bool = True, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all fragments from all file types.
        
        Args:
            normalize: If True, normalize pixel values to [0, 1]
            verbose: If True, print loading progress
            
        Returns:
            Tuple of (X, y, file_types) where:
                X: Fragment data array of shape (n_samples, fragment_size)
                y: Encoded labels array of shape (n_samples,)
                file_types: List of file type names corresponding to label indices
        """
        self.file_types = self.get_file_types()
        
        if verbose:
            print(f"Loading fragments for {len(self.file_types)} file types...")
            print(f"File types: {', '.join(self.file_types)}")
            print()
        
        all_fragments = []
        all_labels = []
        
        for file_type in self.file_types:
            if verbose:
                print(f"Loading {file_type}...", end=' ')
            
            try:
                fragments, names = self.load_fragments(file_type)
                
                if len(fragments) > 0:
                    all_fragments.append(fragments)
                    all_labels.extend([file_type] * len(fragments))
                    
                    if verbose:
                        print(f"✅ {len(fragments):,} fragments")
                else:
                    if verbose:
                        print(f"⚠️  No valid fragments")
                        
            except Exception as e:
                if verbose:
                    print(f"❌ Error: {e}")
        
        if not all_fragments:
            raise ValueError("No fragments loaded!")
        
        # Concatenate all fragments
        X = np.vstack(all_fragments)
        
        # Encode labels
        y = self.label_encoder.fit_transform(all_labels)
        
        # Normalize if requested
        if normalize:
            X = X.astype(np.float32) / 255.0
        
        if verbose:
            print()
            print(f"Total fragments loaded: {len(X):,}")
            print(f"Fragment shape: {X.shape}")
            print(f"Unique labels: {len(self.file_types)}")
            print()
        
        return X, y, self.file_types
    
    def save_label_encoder(self, filepath: str):
        """Save label encoder to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to {filepath}")
    
    def load_label_encoder(self, filepath: str):
        """Load label encoder from file."""
        with open(filepath, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"Label encoder loaded from {filepath}")
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get distribution of samples per class."""
        distribution = {}
        for label in np.unique(y):
            class_name = self.label_encoder.inverse_transform([label])[0]
            count = np.sum(y == label)
            distribution[class_name] = count
        return distribution


def load_train_data(train_dir: str = "Train", 
                     normalize: bool = True, 
                     verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], FragmentDataLoader]:
    """
    Convenience function to load training data.
    
    Args:
        train_dir: Directory containing fragment subdirectories
        normalize: If True, normalize to [0, 1]
        verbose: If True, print progress
        
    Returns:
        Tuple of (X, y, file_types, loader)
    """
    loader = FragmentDataLoader(train_dir)
    X, y, file_types = loader.load_all(normalize=normalize, verbose=verbose)
    
    if verbose:
        distribution = loader.get_class_distribution(y)
        print("Class distribution:")
        for class_name in sorted(distribution.keys()):
            count = distribution[class_name]
            percentage = (count / len(y)) * 100
            print(f"  {class_name:15s}: {count:7,} ({percentage:5.2f}%)")
        print()
    
    return X, y, file_types, loader


if __name__ == '__main__':
    # Test the data loader
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("=" * 80)
    print("TESTING DATA LOADER")
    print("=" * 80)
    print()
    
    # Load data
    X, y, file_types, loader = load_train_data(
        train_dir=r"c:\Users\prath\Desktop\file-type-identification\Train",
        normalize=True,
        verbose=True
    )
    
    print("=" * 80)
    print("DATA LOADED SUCCESSFULLY!")
    print("=" * 80)
