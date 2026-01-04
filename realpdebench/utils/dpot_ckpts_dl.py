"""
Download DPOT pre-trained models from Hugging Face.

This script downloads all available DPOT checkpoints from the official repository:
https://huggingface.co/hzk17/DPOT/tree/main
"""

import os
from huggingface_hub import hf_hub_download
import argparse

# Available DPOT models with sizes
DPOT_MODELS = {
    'model_Ti.pth': '90.5 MB',    # Tiny
    'model_S.pth': '370 MB',      # Small  
    'model_M.pth': '1.47 GB',     # Medium
    'model_L.pth': '6.11 GB',     # Large
    'model_H.pth': '12.4 GB',     # Huge
}

def download_dpot_models(output_dir="dpot_ckpts", models=None):
    """
    Download DPOT models from Hugging Face.
    
    Args:
        output_dir: Directory to save models
        models: List of specific models to download (None = all)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If no specific models specified, download all
    if models is None:
        models = list(DPOT_MODELS.keys())
    
    print(f"Downloading DPOT models to: {os.path.abspath(output_dir)}")
    print("="*60)
    
    for model_name in models:
        if model_name not in DPOT_MODELS:
            print(f"❌ Unknown model: {model_name}")
            continue
            
        model_size = DPOT_MODELS[model_name]
        output_path = os.path.join(output_dir, model_name)
        
        # Check if model already exists
        if os.path.exists(output_path):
            print(f"✓ {model_name} ({model_size}) - Already exists, skipping")
            continue
        
        try:
            print(f"⬇️  Downloading {model_name} ({model_size})...")
            
            # Download from Hugging Face
            downloaded_path = hf_hub_download(
                repo_id="hzk17/DPOT",
                filename=model_name,
                cache_dir=None,  # Don't use cache, download directly
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ {model_name} downloaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
    
    print("\n" + "="*60)
    print("Download completed!")
    print(f"Models saved in: {os.path.abspath(output_dir)}")
    
    # List downloaded files
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
        if files:
            print(f"\nDownloaded models ({len(files)}):")
            for f in sorted(files):
                size = os.path.getsize(os.path.join(output_dir, f))
                size_str = f"{size / (1024**3):.2f} GB" if size > 1024**3 else f"{size / (1024**2):.1f} MB"
                print(f"  - {f} ({size_str})")
        else:
            print("\nNo .pth files found in output directory")


def main():
    parser = argparse.ArgumentParser(description="Download DPOT pre-trained models")
    parser.add_argument("--output-dir", required=True, 
                        help="Output directory for models")
    parser.add_argument("--models", nargs="+", choices=list(DPOT_MODELS.keys()),
                        help="Specific models to download (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available models and exit")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available DPOT models:")
        print("="*40)
        for model, size in DPOT_MODELS.items():
            print(f"  {model:<15} ({size})")
        return
    
    download_dpot_models(args.output_dir, args.models)


if __name__ == "__main__":
    main() 