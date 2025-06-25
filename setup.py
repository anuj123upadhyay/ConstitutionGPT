#!/usr/bin/env python3
"""
Setup script for Constitution RAG system
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.helpers import validate_environment_variables, print_validation_results

def setup_environment():
    """Setup and validate environment"""
    print("Setting up Constitution RAG System...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    
    # Validate environment variables
    validation_results = validate_environment_variables()
    print_validation_results(validation_results)
    
    if not all(validation_results.values()):
        print("\nPlease set all required environment variables in your .env file")
        print("Copy .env.example to .env and fill in your API keys")
        return False
    
    # Check for constitution document
    constitution_paths = [
        "documents/COI.pdf",
    
    ]
    
    constitution_found = False
    for path in constitution_paths:
        if os.path.exists(path):
            constitution_found = True
            print(f"Found constitution file: {path}")
            break
    
    if not constitution_found:
        print(f"\nWarning: Constitution file not found!")
        print("Please add the Constitution of India file (PDF or TXT) to the documents folder with one of these names:")
        for path in constitution_paths:
            print(f"- {path}")
        return False
    
    print("\n✓ Environment setup complete!")
    print("Run: streamlit run src/app.py")
    return True

if __name__ == "__main__":
    setup_environment()
