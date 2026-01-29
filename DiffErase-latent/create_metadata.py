#!/usr/bin/env python3
"""
Script to create metadata files for LibriSpeech dataset
Train: most files, Val: 100 files, Test: 100 files
"""
import os
import json
import random

def find_wav_files(directory):
    """Find all wav files in directory"""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                # Get relative path from directory
                rel_path = os.path.relpath(full_path, directory)
                wav_files.append(rel_path)
    return sorted(wav_files)

def create_metadata_json(wav_files, output_path):
    """Create metadata JSON file"""
    data = []
    for wav_file in wav_files:
        # For unconditional generation, we don't need text or labels
        data.append({
            "wav": wav_file,
            "text": "",  # Empty text for unconditional
            "labels": ""  # No labels needed
        })
    
    metadata = {
        "data": data
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created metadata file: {output_path}")
    print(f"Total files: {len(data)}")

def main():
    librispeech_root = "/path/to/LibriSpeech/train-clean-100"
    metadata_dir = "data/dataset/metadata/librispeech"
    
    # Create metadata directory
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Find all wav files
    print(f"Scanning {librispeech_root} for wav files...")
    all_wav_files = find_wav_files(librispeech_root)
    print(f"Found {len(all_wav_files)} wav files")
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(all_wav_files)
    
    # Split: val=100, test=100, rest for train
    n_val = 100
    n_test = 100
    
    val_files = all_wav_files[:n_val]
    test_files = all_wav_files[n_val:n_val+n_test]
    train_files = all_wav_files[n_val+n_test:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")
    
    # Create metadata files
    create_metadata_json(train_files, os.path.join(metadata_dir, "train.json"))
    create_metadata_json(val_files, os.path.join(metadata_dir, "val.json"))
    create_metadata_json(test_files, os.path.join(metadata_dir, "test.json"))
    
    # Create or update dataset_root.json entry
    dataset_root_path = "data/dataset/metadata/dataset_root.json"
    if os.path.exists(dataset_root_path):
        with open(dataset_root_path, 'r') as f:
            dataset_root = json.load(f)
    else:
        dataset_root = {}
    
    dataset_root["librispeech"] = librispeech_root
    
    if "metadata" not in dataset_root:
        dataset_root["metadata"] = {}
    if "path" not in dataset_root["metadata"]:
        dataset_root["metadata"]["path"] = {}
    
    dataset_root["metadata"]["path"]["librispeech"] = {
        "train": os.path.join(metadata_dir, "train.json"),
        "val": os.path.join(metadata_dir, "val.json"),
        "test": os.path.join(metadata_dir, "test.json")
    }
    
    with open(dataset_root_path, 'w') as f:
        json.dump(dataset_root, f, indent=2)
    
    print(f"\nUpdated {dataset_root_path}")
    print("Metadata files created successfully!")

if __name__ == "__main__":
    main()

