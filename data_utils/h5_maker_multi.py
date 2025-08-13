#!/usr/bin/env python3
"""
Author: Aritra Bal, ETP
Date: XIII Idibus Sextilibus anno ab urbe condita MMDCCLXXVIII

Process ROOT files containing jet data for particle physics classification.
Converts awkward arrays to structured numpy arrays and saves as HDF5 files.
Multiprocessing enabled for faster processing.
"""

import argparse
import glob
import os
import pdb
import random
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import Pool
import multiprocessing as mp

import pathlib
import awkward as ak
import h5py
import numpy as np
import uproot
from tqdm import tqdm


def calculate_pt(px: ak.Array, py: ak.Array) -> ak.Array:
    """Calculate transverse momentum from px and py components."""
    return np.sqrt(px**2 + py**2)


def pad_and_truncate(array: ak.Array, max_particles: int = 10) -> np.ndarray:
    """
    Pad or truncate awkward array to fixed size.
    
    Args:
        array: Input awkward array
        max_particles: Maximum number of particles to keep
        
    Returns:
        Numpy array of shape (n_jets, max_particles)
    """
    # Convert to numpy and pad/truncate
    padded = ak.fill_none(ak.pad_none(array, max_particles, clip=True), 0.0)
    return ak.to_numpy(padded)


def process_particle_features(tree: uproot.TTree, max_particles: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Process particle-level features from the tree.
    
    Args:
        tree: ROOT tree containing the data
        max_particles: Maximum number of particles per jet
        
    Returns:
        jetConstituentsList: (n_jets, max_particles, 3) array with pt, deta, dphi
        jetConstituentsExtra: (n_jets, max_particles, n_features) array with all other part_ features
        jetConstituentsExtraNames: List of feature names
    """
    # Get particle branches
    part_branches = [key for key in tree.keys() if key.startswith('part_')]
    
    # Calculate pt and get deta, dphi for jetConstituentsList
    part_px = tree['part_px'].array()
    part_py = tree['part_py'].array()
    part_pt = calculate_pt(part_px, part_py)
    part_deta = tree['part_deta'].array()
    part_dphi = tree['part_dphi'].array()
    
    # Create jetConstituentsList (pt, deta, dphi)
    pt_padded = pad_and_truncate(part_pt, max_particles)
    deta_padded = pad_and_truncate(part_deta, max_particles)
    dphi_padded = pad_and_truncate(part_dphi, max_particles)
    
    jetConstituentsList = np.stack([pt_padded, deta_padded, dphi_padded], axis=-1)
    
    # Process all other part_ features for jetConstituentsExtra
    extra_features = []
    extra_names = []
    
    for branch_name in sorted(part_branches):
        if branch_name not in ['part_deta', 'part_dphi']:  # deta and dphi already in jetConstituentsList
            feature_array = tree[branch_name].array()
            padded_feature = pad_and_truncate(feature_array, max_particles)
            extra_features.append(padded_feature)
            extra_names.append(branch_name)
    
    jetConstituentsExtra = np.stack(extra_features, axis=-1)
    
    return jetConstituentsList, jetConstituentsExtra, extra_names


def process_jet_features(tree: uproot.TTree) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Process jet-level features.
    
    Args:
        tree: ROOT tree containing the data
        
    Returns:
        jetFeatures: (n_jets, 4) array with pt, eta, phi, energy
        jetFeaturesNames: List of feature names
        jetExtraFeatures: (n_jets, n_extra) array with remaining jet features
        jetExtraFeaturesNames: List of extra feature names
    """
    # Main jet features
    main_features = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
    jetFeatures = np.column_stack([ak.to_numpy(tree[name].array()) for name in main_features])
    
    # Extra jet features
    jet_branches = [key for key in tree.keys() if key.startswith('jet_') and key not in main_features]
    jetExtraFeatures = np.column_stack([ak.to_numpy(tree[name].array()) for name in sorted(jet_branches)])
    
    return jetFeatures, main_features, jetExtraFeatures, sorted(jet_branches)


def process_truth_labels(tree: uproot.TTree) -> np.ndarray:
    """
    Process truth labels: 0 for QCD, 1 for Tbqq.
    
    Args:
        tree: ROOT tree containing the data
        
    Returns:
        truth_labels: (n_jets,) array with truth labels
    """
    label_qcd = ak.to_numpy(tree['label_QCD'].array())
    label_tbqq = ak.to_numpy(tree['label_Tbqq'].array())
    
    # Create truth labels: 0 for QCD, 1 for Tbqq
    truth_labels = np.where(label_qcd == 1, 0, np.where(label_tbqq == 1, 1, -1))
    
    # Remove invalid labels (-1)
    valid_mask = truth_labels != -1
    if not np.all(valid_mask):
        tqdm.write(f"⚠️  Found {np.sum(~valid_mask)} jets with invalid labels")
    
    return truth_labels


def process_qf_features(tree: uproot.TTree) -> np.ndarray:
    """
    Process qfZZ features and reshape to jetConstituentsQFI.
    
    Args:
        tree: ROOT tree containing the data
        
    Returns:
        jetConstituentsQFI: (n_jets, 30, 30) array
    """
    # Find all qf branches (qf0-qf9, qf10-qf89)
    qf_branches = []
    for i in range(90):
        branch_name = f'qf{i}'
        if branch_name in tree.keys():
            qf_branches.append(branch_name)
    
    if len(qf_branches) != 90:
        raise ValueError(f"Expected 90 qf branches, found {len(qf_branches)}")
    
    # Process each qf branch and stack them
    qf_arrays = []
    for branch_name in sorted(qf_branches, key=lambda x: int(x[2:])):  # Sort numerically
        qf_array = tree[branch_name].array()
        qf_padded = pad_and_truncate(qf_array, max_particles=10)
        qf_arrays.append(qf_padded)
    
    # Stack to create (n_jets, 90, 10) array
    qf_stacked = np.stack(qf_arrays, axis=1)
    
    # Reshape to (n_jets, 30, 30)
    n_jets = qf_stacked.shape[0]
    jetConstituentsQFI = qf_stacked.reshape(n_jets, 30, 30)
    
    return jetConstituentsQFI


def process_single_file(file_path: str) -> Dict[str, np.ndarray]:
    """
    Process a single ROOT file and extract all required features.
    
    Args:
        file_path: Path to the ROOT file
        
    Returns:
        Dictionary containing processed arrays
    """
    with uproot.open(file_path) as root_file:
        # Get the first tree
        tree_names = [key for key in root_file.keys() if root_file[key].classname.startswith('TTree')]
        if not tree_names:
            raise ValueError(f"No TTrees found in {file_path}")
        
        tree = root_file[tree_names[0]]
        
        # Process different feature groups
        jetConstituentsList, jetConstituentsExtra, jetConstituentsExtraNames = process_particle_features(tree)
        jetFeatures, jetFeaturesNames, jetExtraFeatures, jetExtraFeaturesNames = process_jet_features(tree)
        truth_labels = process_truth_labels(tree)
        jetConstituentsQFI = process_qf_features(tree)
        
        return {
            'jetConstituentsList': jetConstituentsList,
            'jetConstituentsExtra': jetConstituentsExtra,
            'jetConstituentsExtraNames': jetConstituentsExtraNames,
            'jetFeatures': jetFeatures,
            'jetFeaturesNames': jetFeaturesNames,
            'jetExtraFeatures': jetExtraFeatures,
            'jetExtraFeaturesNames': jetExtraFeaturesNames,
            'truth_labels': truth_labels,
            'jetConstituentsQFI': jetConstituentsQFI
        }


def combine_files_data(file_data_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Combine data from multiple files.
    
    Args:
        file_data_list: List of dictionaries containing file data
        
    Returns:
        Combined data dictionary
    """
    if not file_data_list:
        raise ValueError("No files to combine")
    
    combined_data = {}
    
    # Combine numerical arrays
    for key in ['jetConstituentsList', 'jetConstituentsExtra', 'jetFeatures', 
                'jetExtraFeatures', 'truth_labels', 'jetConstituentsQFI']:
        arrays = [data[key] for data in file_data_list]
        combined_data[key] = np.concatenate(arrays, axis=0)
    
    # Keep names from first file (they should be the same across files)
    for key in ['jetConstituentsExtraNames', 'jetFeaturesNames', 'jetExtraFeaturesNames']:
        combined_data[key] = file_data_list[0][key]
    
    return combined_data


def save_to_hdf5(data: Dict[str, np.ndarray], output_path: str) -> None:
    """
    Save processed data to HDF5 file.
    
    Args:
        data: Dictionary containing processed arrays
        output_path: Path to output HDF5 file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as h5f:
        # Save numerical arrays
        for key in ['jetConstituentsList', 'jetConstituentsExtra', 'jetFeatures', 
                    'jetExtraFeatures', 'truth_labels', 'jetConstituentsQFI']:
            h5f.create_dataset(key, data=data[key], compression='gzip')
        
        # Save string arrays (feature names)
        for key in ['jetConstituentsExtraNames', 'jetFeaturesNames', 'jetExtraFeaturesNames']:
            h5f.create_dataset(key, data=[name.encode('utf-8') for name in data[key]])


def process_group(args_tuple: Tuple[str, List[str], str, str, bool]) -> Tuple[str, bool]:
    """
    Process a single group of files.
    
    Args:
        args_tuple: (xxx, group_files, data_type, output_dir, trial_run)
        
    Returns:
        Tuple of (xxx, success)
    """
    xxx, group_files, data_type, output_dir, trial_run = args_tuple
    
    try:
        # Process all files in the group
        file_data_list = []
        
        # Create progress bar for files in this group
        file_pbar = tqdm(
            sorted(group_files), 
            desc=f"📁 Group {xxx}",
            leave=False,
            colour='green'
        )
        
        for file_path in file_pbar:
            try:
                file_data = process_single_file(file_path)
                file_data_list.append(file_data)
                file_pbar.set_postfix({'✅': os.path.basename(file_path)[:20]})
            except Exception as e:
                tqdm.write(f"❌ Error processing {os.path.basename(file_path)}: {e}")
                continue
        
        file_pbar.close()
        
        if not file_data_list:
            tqdm.write(f"❌ No valid files processed for group {xxx}")
            return xxx, False
        
        # Combine data from all files in the group
        combined_data = combine_files_data(file_data_list)
        
        # Save to HDF5
        output_path = os.path.join(output_dir, f"{data_type}_{xxx}.h5")
        
        if trial_run:
            tqdm.write(f"\n🔍 Trial run complete for group {xxx}. Data shapes:")
            for key, value in combined_data.items():
                if isinstance(value, np.ndarray):
                    tqdm.write(f"  {key}: {value.shape}")
                else:
                    tqdm.write(f"  {key}: {len(value)} items")
            
            # Set global variables for debugging
            globals()['combined_data'] = combined_data
            globals()['output_path'] = output_path
            
            tqdm.write(f"\n🐛 Entering debug mode for group {xxx}...")
            tqdm.write(f"Available variables: 'combined_data', 'output_path'")
            tqdm.write(f"To save the file, run: save_to_hdf5(combined_data, output_path)")
            pdb.set_trace()
        else:
            save_to_hdf5(combined_data, output_path)
            
        return xxx, True
        
    except Exception as e:
        tqdm.write(f"❌ Error processing group {xxx}: {e}")
        return xxx, False


def process_data_type(input_dir: str, output_dir: str, data_type: str, purpose: str, trial_run: bool = False) -> None:
    """
    Process all files of a specific data type and purpose using multiprocessing.
    
    Args:
        input_dir: Base input directory
        data_type: Either 'ZJetsToNuNu' or 'TTBar'
        purpose: Either 'train' or 'val'
        trial_run: If True, process only one random file for testing
    """
    # Construct input path
    input_path = os.path.join(input_dir, purpose)
    pattern = os.path.join(input_path, f"LUND_{data_type}_*.root")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No files found matching pattern: {pattern}")
        return
    
    print(f"📂 Found {len(files)} files for {data_type} {purpose}")
    
    if trial_run:
        # Select one random file for trial
        files = [random.choice(files)]
        print(f"🧪 Trial run: processing only {os.path.basename(files[0])}")
    
    # Group files by XXX (the three-digit number)
    file_groups = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract XXX from LUND_TYPE_XXX_YY.root
        parts = filename.replace('.root', '').split('_')
        xxx = parts[2]  # The XXX part
        
        if xxx not in file_groups:
            file_groups[xxx] = []
        file_groups[xxx].append(file_path)
    
    print(f"🔀 Processing {len(file_groups)} groups with multiprocessing")
    
    # Determine number of processes
    max_processes = min(12, len(file_groups))
    print(f"🚀 Using {max_processes} processes")
    
    # Prepare arguments for multiprocessing
    process_args = [
        (xxx, group_files, data_type, output_dir, trial_run)
        for xxx, group_files in file_groups.items()
    ]
    
    if trial_run:
        # For trial run, process sequentially to allow debugging
        for args in process_args:
            process_group(args)
    else:
        # Use multiprocessing for normal runs
        with Pool(processes=max_processes) as pool:
            # Create progress bar for groups
            with tqdm(total=len(file_groups), desc="🏗️  Processing Groups", colour='blue') as pbar:
                results = []
                for args in process_args:
                    result = pool.apply_async(process_group, (args,))
                    results.append(result)
                
                # Monitor progress
                for result in results:
                    xxx, success = result.get()
                    status = "✅" if success else "❌"
                    pbar.set_postfix({'Group': xxx, 'Status': status})
                    pbar.update(1)
        
        # Summary
        successful_groups = sum(1 for result in [r.get() for r in results] if result[1])
        print(f"\n🎯 Completed: {successful_groups}/{len(file_groups)} groups processed successfully")


def main() -> None:
    """Main function to parse arguments and process data."""
    parser = argparse.ArgumentParser(
        description="Process ROOT files for jet classification with multiprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--type", 
        choices=["qcd", "ttbar"], 
        default="qcd",
        help="Data type: qcd (ZJetsToNuNu) or ttbar (TTBar)"
    )
    
    parser.add_argument(
        "--purpose", 
        choices=["train", "val"], 
        default="train",
        help="Data purpose: train or val"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/ceph/bmaier/ParT/hybrid",
        help="Base input directory containing PURPOSE subdirectories"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/ceph/abal/QML/qGNN",
        help="Base output directory containing PURPOSE subdirectories"
    )
    
    parser.add_argument(
        "--trial-run",
        action="store_true",
        help="Process only one random file and enter debug mode"
    )
    
    args = parser.parse_args()
    
    # Map user input to actual data type names
    data_type_map = {
        "qcd": "ZJetsToNuNu",
        "ttbar": "TTBar"
    }
    
    data_type = data_type_map[args.type]

    pathlib.Path(os.path.join(args.output_dir,args.purpose)).mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Processing {data_type} data for {args.purpose}")
    print(f"📥 Input directory: {args.input_dir}")
    print(f"📤 Output directory: {os.path.join(args.output_dir,args.purpose)}")
    
    try:
        process_data_type(args.input_dir, os.path.join(args.output_dir,args.purpose), data_type, args.purpose, args.trial_run)
        print("\n🎉 Processing completed successfully!")
    except Exception as e:
        print(f"💥 Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Better for multiprocessing with large data
    exit(main())