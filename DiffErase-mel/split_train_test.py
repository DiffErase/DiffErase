"""
Split a directory of audio files into train/test indices. Writes JSON only (no file moves).
Default: 100 random files as test, rest as train. Used before building mel dataset.
"""

import random
import json
import argparse
from pathlib import Path

def split_dataset(source_dir, output_dir=None, test_count=100, seed=42):
    """
    Randomly select test_count files for test set, rest for train set.
    Only generates JSON files with file paths, does not move files.
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    # Use source_dir as output_dir if not specified
    if output_dir is None:
        output_path = source_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect audio files
    print("Collecting audio files...")
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(source_path.rglob(ext))
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {source_dir}")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Random shuffle
    random.seed(seed)
    random.shuffle(audio_files)
    
    # Split: first test_count for test, rest for train
    if test_count > len(audio_files):
        raise ValueError(f"Test count ({test_count}) exceeds total files ({len(audio_files)})")
    
    test_files = audio_files[:test_count]
    train_files = audio_files[test_count:]
    
    print(f"Train set: {len(train_files)} files")
    print(f"Test set: {len(test_files)} files")
    
    # Prepare JSON data
    train_data = {"count": len(train_files), "files": []}
    test_data = {"count": len(test_files), "files": []}
    
    # Collect file paths (no moving)
    print("\nCollecting file paths...")
    for file_path in train_files:
        relative_path = file_path.relative_to(source_path)
        train_data["files"].append({
            "path": str(file_path),
            "relative_path": str(relative_path)
        })
    
    for file_path in test_files:
        relative_path = file_path.relative_to(source_path)
        test_data["files"].append({
            "path": str(file_path),
            "relative_path": str(relative_path)
        })
    
    # Save JSON files
    print("\nSaving JSON files...")
    train_json_path = output_path / "train_files.json"
    test_json_path = output_path / "test_files.json"
    split_info_path = output_path / "split_info.json"
    
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # Save split info
    split_info = {
        "total_files": len(audio_files),
        "train_count": len(train_files),
        "test_count": len(test_files),
        "random_seed": seed,
        "source_dir": str(source_path)
    }
    
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Done!")
    print(f"JSON files saved to: {output_path}")
    print(f"  - Train files: {train_json_path}")
    print(f"  - Test files: {test_json_path}")
    print(f"  - Split info: {split_info_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split audio files into train/test sets (JSON only)")
    parser.add_argument("source_dir", type=str, help="Source directory containing audio files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for JSON files (default: same as source_dir)")
    parser.add_argument("--test_count", type=int, default=100, help="Number of test samples (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Audio Dataset Train/Test Split (JSON only)")
    print("=" * 60)
    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir or args.source_dir}")
    print(f"Test set: {args.test_count} random samples")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()
    
    split_dataset(args.source_dir, args.output_dir, test_count=args.test_count, seed=args.seed)
