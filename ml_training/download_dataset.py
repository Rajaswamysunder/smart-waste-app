#!/usr/bin/env python3
"""
Download Waste Classification Datasets
======================================
Downloads public waste classification datasets for training.

Sources:
1. TrashNet (Gary Thung) - 6 classes, 2527 images
2. Additional augmentation with your own images

Usage:
    python download_dataset.py
"""

import os
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Dataset URLs
TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"

# Mapping TrashNet classes to our classes
CLASS_MAPPING = {
    'cardboard': 'recyclable',
    'glass': 'recyclable', 
    'metal': 'recyclable',
    'paper': 'recyclable',
    'plastic': 'recyclable',
    'trash': 'general'
}

def download_with_progress(url, filename):
    """Download file with progress bar."""
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)

def download_trashnet():
    """Download and extract TrashNet dataset."""
    
    print("\n" + "="*60)
    print("📥 DOWNLOADING TRASHNET DATASET")
    print("="*60)
    
    # Create temp directory
    temp_dir = Path('temp_download')
    temp_dir.mkdir(exist_ok=True)
    
    zip_path = temp_dir / 'trashnet.zip'
    
    # Download
    print(f"\n📥 Downloading from GitHub...")
    print(f"   URL: {TRASHNET_URL}")
    
    try:
        download_with_progress(TRASHNET_URL, str(zip_path))
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://github.com/garythung/trashnet")
        return False
    
    # Extract
    print("\n📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find extracted folder
    extracted_dir = temp_dir / 'dataset-resized'
    if not extracted_dir.exists():
        # Try to find it
        for item in temp_dir.iterdir():
            if item.is_dir() and item.name != '__MACOSX':
                extracted_dir = item
                break
    
    print(f"   Extracted to: {extracted_dir}")
    
    # Create our dataset structure
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True)
    
    for our_class in ['organic', 'recyclable', 'hazardous', 'ewaste', 'general']:
        (dataset_dir / our_class).mkdir(exist_ok=True)
    
    # Copy and organize images
    print("\n📂 Organizing images...")
    
    total_copied = 0
    for trashnet_class, our_class in CLASS_MAPPING.items():
        source_dir = extracted_dir / trashnet_class
        if source_dir.exists():
            dest_dir = dataset_dir / our_class
            
            images = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
            for img in images:
                # Rename to avoid conflicts
                new_name = f"{trashnet_class}_{img.name}"
                shutil.copy2(img, dest_dir / new_name)
                total_copied += 1
            
            print(f"   ✅ {trashnet_class} → {our_class}: {len(images)} images")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ Downloaded {total_copied} images!")
    return True

def create_organic_hazardous_ewaste():
    """
    Create placeholder datasets for missing classes.
    These need manual collection or augmentation.
    """
    
    print("\n" + "="*60)
    print("⚠️  MISSING CLASSES")
    print("="*60)
    
    missing = ['organic', 'hazardous', 'ewaste']
    
    print("\nTrashNet doesn't include these classes:")
    for cls in missing:
        print(f"   • {cls}")
    
    print("\n📷 You need to collect images for these classes:")
    print("""
    ORGANIC (500+ images needed):
    - Food waste, fruit peels, vegetable scraps
    - Leaves, grass clippings, garden waste
    - Coffee grounds, tea bags
    - Eggshells, nut shells
    
    HAZARDOUS (500+ images needed):
    - Batteries (AA, AAA, phone batteries)
    - Paint cans, chemical bottles
    - Cleaning product containers
    - Medicine bottles, syringes
    - Pesticide containers
    
    E-WASTE (500+ images needed):
    - Old phones, tablets
    - Cables, chargers, headphones
    - Computer parts, keyboards
    - Remote controls
    - Broken electronics
    """)
    
    print("💡 Tips for collecting images:")
    print("   1. Take photos with your phone")
    print("   2. Use varied backgrounds and lighting")
    print("   3. Include different angles")
    print("   4. Mix indoor and outdoor shots")
    print("   5. Search Kaggle for additional datasets")

def show_dataset_stats():
    """Show current dataset statistics."""
    
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    
    dataset_dir = Path('dataset')
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found!")
        return
    
    classes = ['organic', 'recyclable', 'hazardous', 'ewaste', 'general']
    total = 0
    
    print("\n" + "-"*40)
    for cls in classes:
        cls_dir = dataset_dir / cls
        if cls_dir.exists():
            images = list(cls_dir.glob('*.jpg')) + \
                    list(cls_dir.glob('*.jpeg')) + \
                    list(cls_dir.glob('*.png'))
            count = len(images)
            total += count
            
            status = "✅" if count >= 500 else "⚠️" if count >= 100 else "❌"
            bar = "█" * min(count // 50, 20)
            print(f"{status} {cls:15} : {count:5} images  {bar}")
        else:
            print(f"❌ {cls:15} : 0 images (folder missing)")
    
    print("-"*40)
    print(f"   {'TOTAL':15} : {total:5} images")
    
    if total >= 2500:
        print("\n✅ Dataset is ready for training!")
        print("   Run: python train_model.py")
    else:
        print("\n⚠️  More images needed for good results.")
        print("   Recommended: 500+ per class")

def main():
    print("\n" + "="*60)
    print("🗑️  WASTE DATASET DOWNLOADER")
    print("="*60)
    
    print("\nOptions:")
    print("1. Download TrashNet dataset")
    print("2. Show dataset statistics")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        success = download_trashnet()
        if success:
            create_organic_hazardous_ewaste()
        show_dataset_stats()
    elif choice == '2':
        show_dataset_stats()
    else:
        print("Goodbye!")

if __name__ == '__main__':
    main()
