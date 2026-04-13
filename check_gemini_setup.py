#!/usr/bin/env python3
"""
check_gemini_setup.py - Verify Gemini API is properly configured
Run this before starting the Streamlit app to diagnose issues
"""

import os
import sys
from pathlib import Path

def check_gemini_key():
    """Check if GEMINI_API_KEY is set."""
    print("🔍 Checking Gemini API Setup...")
    print("=" * 50)
    
    # Check environment
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    
    if not gemini_key:
        print("❌ GEMINI_API_KEY is NOT set\n")
        print("📋 Setup Instructions:")
        print("=" * 50)
        print("\n1️⃣ Get your free API key at:")
        print("   https://ai.google.dev/\n")
        
        print("2️⃣ Set the environment variable (pick ONE method):\n")
        
        print("   Method A - Terminal (temporary):")
        print("   $ export GEMINI_API_KEY='your_api_key_here'")
        print("   $ streamlit run app.py\n")
        
        print("   Method B - Create .env file in the app folder:")
        print("   $ nano .env")
        print("   # Add this line:")
        print("   GEMINI_API_KEY=your_api_key_here")
        print("   # Save and exit (Ctrl+X, Y, Enter)")
        print("   $ streamlit run app.py\n")
        
        print("   Method C - Add to ~/.bash_profile or ~/.zshrc:")
        print("   echo 'export GEMINI_API_KEY=\"your_api_key_here\"' >> ~/.zshrc")
        print("   source ~/.zshrc\n")
        
        return False
    else:
        print(f"✅ GEMINI_API_KEY is SET")
        print(f"   • Key length: {len(gemini_key)} characters")
        print(f"   • First 8 chars: {gemini_key[:8]}...")
        print(f"   • Looks valid: {'AIza' in gemini_key[:20]}\n")
        return True


def check_dependencies():
    """Check if required packages are installed."""
    print("📦 Checking Dependencies...")
    print("=" * 50)
    
    dependencies = {
        "streamlit": "UI Framework",
        "google": "Google GenAI",
        "torch": "PyTorch",
        "torchvision": "PyTorch Vision",
        "PIL": "Pillow",
    }
    
    all_installed = True
    for package, desc in dependencies.items():
        try:
            __import__(package)
            print(f"  ✅ {package:20} ({desc})")
        except ImportError:
            print(f"  ❌ {package:20} ({desc}) - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n⚠️  Install missing packages with:")
        print("   pip install -r requirements.txt")
    
    print()
    return all_installed


def check_model_weights():
    """Check if model weights are available."""
    print("🤖 Checking Model Weights...")
    print("=" * 50)
    
    weights_path = Path("outputs/best_model.pt")
    
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Model weights found")
        print(f"     • Path: {weights_path}")
        print(f"     • Size: {size_mb:.2f} MB\n")
        return True
    else:
        print(f"  ⚠️  Model weights NOT found")
        print(f"     • Expected at: {weights_path}")
        print(f"     • Run: python trainer.py\n")
        return False


def check_data():
    """Check if training data is available."""
    print("📂 Checking Data...")
    print("=" * 50)
    
    data_path = Path("data")
    
    if data_path.exists():
        subfolders = [d for d in data_path.iterdir() if d.is_dir()]
        print(f"  ✅ Data folder exists")
        print(f"     • Subfolders: {', '.join(d.name for d in subfolders)}")
        
        # Count images
        image_count = 0
        for img_path in data_path.rglob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_count += 1
        
        if image_count > 0:
            print(f"     • Total images: {image_count}\n")
        else:
            print(f"     • ⚠️  No images found - Run: python prepare_data.py\n")
        
        return True
    else:
        print(f"  ⚠️  Data folder NOT found")
        print(f"     • Run: python prepare_data.py\n")
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 50)
    print("RadVision AI - Setup Verification")
    print("=" * 50 + "\n")
    
    api_ok = check_gemini_key()
    deps_ok = check_dependencies()
    weights_ok = check_model_weights()
    data_ok = check_data()
    
    print("=" * 50)
    print("📊 Summary")
    print("=" * 50)
    
    status = {
        "✅ Gemini API Key": api_ok,
        "✅ Dependencies": deps_ok,
        "⚠️  Model Weights": weights_ok,
        "⚠️  Training Data": data_ok,
    }
    
    for check, result in status.items():
        symbol = "✅" if result else "❌"
        print(f"{symbol} {check}: {'Ready' if result else 'Missing'}")
    
    print("=" * 50)
    
    if api_ok and deps_ok:
        print("\n✨ You're ready to run RadVision AI!")
        print("   $ streamlit run app.py\n")
        return 0
    else:
        print("\n⚠️  Please fix the issues above before running the app\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
