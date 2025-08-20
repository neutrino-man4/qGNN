"""
Author: Aritra Bal, ETP
Date: XVIII Sextilis anno ab urbe condita MMDCCLXXVIII

Merge TTBar and ZJets h5 files while maintaining label correspondence.
Shuffles combined data while preserving feature-label alignment.
"""

import argparse
import glob
import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def merge_files(ttbar_path: str, zjet_path: str, output_path: str) -> None:
    """
    Merge one TTBar and one ZJet file while preserving label correspondence.
    
    Args:
        ttbar_path: Path to TTBar h5 file
        zjet_path: Path to ZJet h5 file  
        output_path: Path to output merged file
    """
    print(f"  Merging: {os.path.basename(ttbar_path)} + {os.path.basename(zjet_path)}")
    
    # Read data from both files
    with h5py.File(ttbar_path, 'r') as ttbar_f, h5py.File(zjet_path, 'r') as zjet_f:
        dataset_names = list(ttbar_f.keys())
        merged_data = {}
        #import pdb;pdb.set_trace()
        
        # Process each dataset
        for name in dataset_names:
            ttbar_data = ttbar_f[name][:]
            zjet_data = zjet_f[name][:]
            
            # Handle string datasets (feature names) - use TTBar version
            # check if the string 'Names' (case-insensitive) is present in the dtype name
            if 'Names' in name:
                merged_data[name] = ttbar_data
                print(f"    {name}: Using string data from TTBar file")
            else:
                # Numerical datasets - concatenate TTBar first, then ZJet
                merged_data[name] = np.concatenate([ttbar_data, zjet_data], axis=0)
                print(f"    {name}: {ttbar_data.shape} + {zjet_data.shape} = {merged_data[name].shape}")
    
    # Verify label distribution before shuffling
    labels = merged_data['truth_labels']
    n_ttbar = np.sum(labels == 1)
    n_zjet = np.sum(labels == 0)
    print(f"    Labels before shuffle: {n_ttbar} TTBar (1), {n_zjet} ZJet (0)")
    
    # Shuffle all data while preserving correspondence
    total_jets = len(labels)
    shuffle_indices = np.random.permutation(total_jets)
    
    # Apply same shuffle to all numerical datasets
    for name, data in merged_data.items():
        if not ('Names' in name):  # Skip string datasets
            merged_data[name] = data[shuffle_indices]
        else:
            print(f'skip the key: {name}')
    # Verify labels are still correct after shuffling
    shuffled_labels = merged_data['truth_labels']
    n_ttbar_after = np.sum(shuffled_labels == 1)
    n_zjet_after = np.sum(shuffled_labels == 0)
    print(f"    Labels after shuffle: {n_ttbar_after} TTBar (1), {n_zjet_after} ZJet (0)")
    
    # Save merged file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as out_f:
        for name, data in merged_data.items():
            #if 'Names' in name:
            out_f.create_dataset(name, data=data)
            #else:
            #    out_f.create_dataset(name, data=data, compression='gzip')
    
    print(f"    Saved: {output_path}")


def main():
    """Main function to merge TTBar and ZJet files."""
    parser = argparse.ArgumentParser(
        description="Merge TTBar and ZJet h5 files while preserving label correspondence"
    )
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default='/ceph/abal/QML/qGNN/train',
        help="Input directory containing TTBar and ZJet files"
    )
    
    parser.add_argument(
        "--purpose", 
        choices=["train", "val"], 
        default="train",
        help="Data purpose: train or val"
    )
    
    parser.add_argument(
        "--output-base", 
        type=str, 
        default="/ceph/abal/QML/qGNN/merged",
        help="Base output directory"
    )
    
    args = parser.parse_args()
    
    print(f"🔍 Searching for files in: {args.input_dir}")
    
    # Find all TTBar and ZJet files
    ttbar_pattern = os.path.join(args.input_dir, "TTBar_*.h5")
    zjet_pattern = os.path.join(args.input_dir, "ZJetsToNuNu_*.h5")
    
    ttbar_files = sorted(glob.glob(ttbar_pattern))
    zjet_files = sorted(glob.glob(zjet_pattern))
    
    print(f"📁 Found {len(ttbar_files)} TTBar files")
    print(f"📁 Found {len(zjet_files)} ZJet files")
    
    # Extract file numbers to match pairs
    ttbar_numbers = {os.path.basename(f).split('_')[1].split('.')[0]: f for f in ttbar_files}
    zjet_numbers = {os.path.basename(f).split('_')[1].split('.')[0]: f for f in zjet_files}
    
    # Find matching pairs
    matching_numbers = set(ttbar_numbers.keys()) & set(zjet_numbers.keys())
    
    if not matching_numbers:
        print("❌ No matching file pairs found!")
        return
    
    print(f"✅ Found {len(matching_numbers)} matching file pairs")
    
    # Create output directory
    output_dir = os.path.join(args.output_base, args.purpose)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📤 Output directory: {output_dir}")
    
    # Set random seed for reproducible shuffling
    np.random.seed(42)
    
    # Process each matching pair
    for file_num in tqdm(sorted(matching_numbers), desc="🔄 Merging files"):
        ttbar_file = ttbar_numbers[file_num]
        zjet_file = zjet_numbers[file_num]
        output_file = os.path.join(output_dir, f"TTBar+ZJets_{file_num}.h5")
        
        try:
            merge_files(ttbar_file, zjet_file, output_file)
        except Exception as e:
            print(f"❌ Error merging {file_num}: {e}")
            continue
    
    print(f"\n🎉 Completed! Merged files saved to: {output_dir}")


if __name__ == "__main__":
    main()