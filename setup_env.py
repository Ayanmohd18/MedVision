#!/usr/bin/env python3
"""
Setup environment variables for the AI Radiology System
"""

import os
import sys

def setup_openai_key():
    """Setup OpenAI API key"""
    print("OpenAI API Key Setup")
    print("=" * 30)
    
    current_key = os.getenv("OPENAI_API_KEY")
    if current_key:
        print(f"Current API key: {current_key[:10]}...{current_key[-4:] if len(current_key) > 14 else current_key}")
        
        choice = input("Update existing key? (y/n): ").lower()
        if choice != 'y':
            return
    
    print("\nEnter your OpenAI API key:")
    print("(Get it from: https://platform.openai.com/api-keys)")
    
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("No API key provided")
        return
    
    if not api_key.startswith("sk-"):
        print("Warning: API key should start with 'sk-'")
    
    # Set for current session
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Create .env file
    with open(".env", "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print(f"\nAPI key set successfully!")
    print("- Current session: ✓")
    print("- .env file: ✓")
    
    if sys.platform == "win32":
        print(f"\nTo set permanently on Windows, run:")
        print(f'setx OPENAI_API_KEY "{api_key}"')
    else:
        print(f"\nTo set permanently on Linux/Mac, add to ~/.bashrc:")
        print(f'export OPENAI_API_KEY="{api_key}"')

def verify_setup():
    """Verify environment setup"""
    print("\nVerifying setup...")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✓ OpenAI API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else api_key}")
    else:
        print("✗ OpenAI API key not found")
    
    # Check .env file
    if os.path.exists(".env"):
        print("✓ .env file created")
    else:
        print("✗ .env file not found")

if __name__ == "__main__":
    setup_openai_key()
    verify_setup()