import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "ai_service"))

try:
    from training.dataset import load_all_entries
    from training.config import CONFIG
    
    print("Successfully imported load_all_entries and CONFIG.")
    
    # Temporarily point config roots to our newly generated data directory
    CONFIG.dataset.FACEFORENSICS_ROOT = str(PROJECT_ROOT / "data" / "FaceForensics")
    CONFIG.dataset.DFDC_ROOT = str(PROJECT_ROOT / "data" / "DFDC")
    CONFIG.dataset.CELEB_DF_ROOT = str(PROJECT_ROOT / "data" / "CelebDF")
    CONFIG.dataset.CUSTOM_ROOT = str(PROJECT_ROOT / "data" / "custom_reels")
    
    print("\nConfigured Paths:")
    print(f"FaceForensics: {CONFIG.dataset.FACEFORENSICS_ROOT}")
    print(f"DFDC: {CONFIG.dataset.DFDC_ROOT}")
    print(f"CelebDF: {CONFIG.dataset.CELEB_DF_ROOT}")
    print(f"Custom: {CONFIG.dataset.CUSTOM_ROOT}")
    
    # Load all entries
    print("\nLoading entries...")
    entries = load_all_entries()
    
    print(f"\nTotal entries loaded: {len(entries)}")
    
    # Group by dataset source
    sources = {}
    for entry in entries:
        src = entry.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        
    print("\nBreakdown by Dataset Source:")
    for src, count in sources.items():
        print(f"  {src}: {count} files")
        
    # Group by label
    labels = {0: "REAL", 1: "FAKE"}
    label_counts = {}
    for entry in entries:
        lbl = entry.get("label", -1)
        label_name = labels.get(lbl, f"unknown ({lbl})")
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
    print("\nBreakdown by Label:")
    for lbl, count in label_counts.items():
        print(f"  {lbl}: {count} files")
        
    print("\nStatus: SUCCESS - Dataset loaders loaded the sample files without errors!")
    
except Exception as e:
    print(f"\nStatus: FAILURE - An error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
