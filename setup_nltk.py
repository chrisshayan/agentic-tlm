#!/usr/bin/env python3
"""
NLTK Data Setup Script - SSL Certificate Workaround
Handles SSL certificate issues on macOS when downloading NLTK data.
"""

import ssl
import sys
import os
from pathlib import Path

def setup_nltk_data():
    """Download NLTK data with SSL certificate workaround."""
    print("🔧 Setting up NLTK data with SSL certificate workaround...")
    
    try:
        # Create unverified HTTPS context for SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python versions
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        import nltk
        
        # Create NLTK data directory
        nltk_data_dir = Path.home() / 'nltk_data'
        nltk_data_dir.mkdir(exist_ok=True)
        
        # Download required datasets
        datasets = [
            'punkt',
            'stopwords', 
            'wordnet',
            'averaged_perceptron_tagger',
            'vader_lexicon'
        ]
        
        for dataset in datasets:
            try:
                print(f"📥 Downloading {dataset}...")
                nltk.download(dataset, quiet=True)
                print(f"✅ {dataset} downloaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not download {dataset}: {e}")
                
                # Try alternative download method
                try:
                    print(f"🔄 Trying alternative download for {dataset}...")
                    nltk.download(dataset, download_dir=str(nltk_data_dir), quiet=True)
                    print(f"✅ {dataset} downloaded via alternative method")
                except Exception as e2:
                    print(f"❌ Failed to download {dataset}: {e2}")
                    continue
        
        print("✅ NLTK data setup completed!")
        
        # Verify installation
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            test_text = "This is a test sentence."
            tokens = word_tokenize(test_text)
            stop_words = set(stopwords.words('english'))
            print(f"🧪 Test successful: {len(tokens)} tokens, {len(stop_words)} stop words")
        except Exception as e:
            print(f"⚠️ Verification failed: {e}")
            
    except ImportError:
        print("❌ NLTK not installed. Please install with: pip install nltk")
        return False
    except Exception as e:
        print(f"❌ Error setting up NLTK: {e}")
        return False
    
    return True

def setup_spacy_model():
    """Download spaCy English model."""
    print("🔧 Setting up spaCy model...")
    
    try:
        import spacy
        
        # Check if model is already installed
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model 'en_core_web_sm' already installed")
            return True
        except OSError:
            pass
        
        # Download model
        print("📥 Downloading spaCy English model...")
        os.system("python -m spacy download en_core_web_sm")
        
        # Verify installation
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model installed successfully")
            return True
        except OSError:
            print("❌ spaCy model installation failed")
            return False
            
    except ImportError:
        print("❌ spaCy not installed. Please install with: pip install spacy")
        return False
    except Exception as e:
        print(f"❌ Error setting up spaCy: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up NLP models and data...")
    print("=" * 50)
    
    success = True
    
    # Setup NLTK data
    if not setup_nltk_data():
        success = False
    
    print("-" * 50)
    
    # Setup spaCy model
    if not setup_spacy_model():
        success = False
    
    print("=" * 50)
    
    if success:
        print("🎉 All NLP components set up successfully!")
        print("✅ You can now use the natural language features.")
    else:
        print("⚠️ Some components failed to install.")
        print("💡 The system will still work with limited NLP capabilities.")
    
    print("\n🔧 Alternative manual setup commands:")
    print("   • For NLTK: python -c \"import nltk; nltk.download('punkt', quiet=True)\"")
    print("   • For spaCy: python -m spacy download en_core_web_sm")
    print("   • Set SSL context: export PYTHONHTTPSVERIFY=0 (temporary)")

if __name__ == "__main__":
    main() 