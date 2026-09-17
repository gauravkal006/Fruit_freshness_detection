import os
import sys
import shutil
from pathlib import Path

# Ensure UTF-8 output encoding for terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def import_colab_weights(source_filepath):
    """
    Imports Google Colab trained model weights (.pt file) into the local project.
    """
    project_dir = Path(__file__).parent.parent
    target_path = project_dir / "yolo11_produce_spoilage_best.pt"
    
    src = Path(source_filepath)
    if not src.exists():
        print(f"❌ Error: File not found at '{source_filepath}'")
        return False
        
    print(f"📥 Copying Google Colab weights from: {src}")
    shutil.copy2(src, target_path)
    
    # Also update cache location
    c_cache_dir = Path(r"C:\Users\HP\.cache\ultralytics\runs\classify\yolo11n_spoilage_15ep\weights")
    c_cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, c_cache_dir / "best.pt")
    
    print("=" * 65)
    print("🎉 GOOGLE COLAB MODEL WEIGHTS IMPORTED SUCCESSFULLY!")
    print(f"  • Local Project Weights: {target_path}")
    print(f"  • Active Cache Weights: {c_cache_dir / 'best.pt'}")
    print("=" * 65)
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        import_colab_weights(sys.argv[1])
    else:
        print("Usage: python FinalCodes/load_colab_model.py <path_to_downloaded_colab_weights.pt>")
