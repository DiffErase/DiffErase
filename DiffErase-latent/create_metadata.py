#!/usr/bin/env python3
"""
Create train/val/test metadata for a folder of .wav files and register it in dataset_root.json.
Train: most files, Val: n_val files, Test: n_test files
"""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_dir", help="Folder containing .wav files (searched recursively)")
    parser.add_argument("--name", default="librispeech", help="Dataset name used in the config `data` entries")
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    args = parser.parse_args()

    audio_root = os.path.abspath(args.audio_dir)
    metadata_dir = f"data/dataset/metadata/{args.name}"
    
    # Create metadata directory
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Find all wav files
    print(f"Scanning {audio_root} for wav files...")
    all_wav_files = find_wav_files(audio_root)
    print(f"Found {len(all_wav_files)} wav files")
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(all_wav_files)
    
    # Split: val=100, test=100, rest for train
    n_val = args.n_val
    n_test = args.n_test
    
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
    
    dataset_root[args.name] = audio_root
    
    if "metadata" not in dataset_root:
        dataset_root["metadata"] = {}
    if "path" not in dataset_root["metadata"]:
        dataset_root["metadata"]["path"] = {}
    
    dataset_root["metadata"]["path"][args.name] = {
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

