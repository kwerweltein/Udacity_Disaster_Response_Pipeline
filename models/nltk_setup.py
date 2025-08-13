import nltk
import os
import pprint

# --- NLTK Data Downloads (Keep this part for actual downloads) ---
# 'punkt' is needed for word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer data...")
    nltk.download('punkt')

# 'punkt_tab' (if still needed, though less common)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK 'punkt_tab' data...")
    nltk.download('punkt_tab')


# 'averaged_perceptron_tagger' is needed for POS tagging (e.g., by WordNetLemmatizer)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK 'averaged_perceptron_tagger' data...")
    nltk.download('averaged_perceptron_tagger')


# 'wordnet' is needed for WordNetLemmatizer
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK 'wordnet' corpus data...")
    nltk.download('wordnet')

# 'omw-1.4' (Open Multilingual Wordnet) is often a dependency for WordNetLemmatizer
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading NLTK 'omw-1.4' corpus data...")
    nltk.download('omw-1.4')

# --- END NLTK Data Downloads ---


# --- Corrected way to list *some* installed packages ---
# This checks for the presence of a few common ones
print("\nChecking status of some common NLTK packages:")
checked_packages = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4',
    'stopwords': 'corpora/stopwords', # Adding another common one
    'words': 'corpora/words' # Another common one
}

installed_status = {}
for name, path in checked_packages.items():
    try:
        nltk.data.find(path)
        installed_status[name] = 'Installed'
    except LookupError:
        installed_status[name] = 'Not Installed'
    except Exception as e:
        installed_status[name] = f'Error checking: {e}'

pprint.pprint(installed_status)

# --- Best way to see all installed (using the GUI or shell) ---
print("\nTo see ALL installable/installed packages, use the NLTK Downloader GUI:")
print(">>> import nltk")
print(">>> nltk.download() # This will open a GUI window")
print("\nOr in shell mode:")
print(">>> nltk.download_shell()")
print("   Then type 'l' (for list) to see all packages and their status.")
#nltk.download()
# --- Original Example Usage (if needed) ---
# if __name__ == "__main__":
#     from nltk.tokenize import word_tokenize
#     from nltk.stem import WordNetLemmatizer
#
#     text = "NLTK is a powerful library for natural language processing."
#
#     # Tokenization
#     tokens = word_tokenize(text)
#     print(f"Original text: {text}")
#     print(f"Tokens: {tokens}")
#
#     # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     lemmas = [lemmatizer.lemmatize(token) for token in tokens]
#     print(f"Lemmas: {lemmas}")
#
#     print("\nNLTK downloads and basic usage successful!")