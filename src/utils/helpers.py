import os
from typing import Dict, Any

def ensure_directory_exists(directory_path: str):
    """Ensure that a directory exists, create if not"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def validate_environment_variables() -> Dict[str, bool]:
    """Validate that all required environment variables are set"""
    required_vars = [
        "PINECONE_API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_MODEL_ENDPOINT"
    ]
    
    validation_results = {}
    for var in required_vars:
        validation_results[var] = bool(os.getenv(var))
    
    return validation_results

def print_validation_results(results: Dict[str, bool]):
    """Print environment variable validation results"""
    print("Environment Variable Validation:")
    print("-" * 40)
    for var, is_set in results.items():
        status = "✓ SET" if is_set else "✗ MISSING"
        print(f"{var}: {status}")
    print("-" * 40)
