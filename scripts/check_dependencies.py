"""
Check if all dependencies are installed correctly
"""

import sys


def check_dependencies():
    """Check required dependencies"""
    
    print("Checking dependencies...")
    print("=" * 50)
    
    required = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "httpx": "HTTPX",
        "pydantic": "Pydantic",
        "sentence_transformers": "SentenceTransformers",
    }
    
    optional = {
        "redis": "Redis",
        "transformers": "Transformers (for NLI model)",
        "torch": "PyTorch (for NLI model)",
    }
    
    missing_required = []
    missing_optional = []
    
    print("\nRequired dependencies:")
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            missing_required.append(name)
    
    print("\nOptional dependencies:")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            missing_optional.append(name)
    
    print("\n" + "=" * 50)
    
    if missing_required:
        print("\nMissing required dependencies:")
        for dep in missing_required:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print("\nMissing optional dependencies:")
        for dep in missing_optional:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements_improved.txt")
    
    print("\nAll required dependencies are installed!")
    return True


if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)

