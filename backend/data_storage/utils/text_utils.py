"""
Text utilities for processing and batching text
"""
import re
import tiktoken
from typing import List, Any

# German-aware stopwords for tokenization
GERMAN_STOPWORDS = {
    "der","die","das","dass","und","oder","nicht","ist","sind","war","waren",
    "ein","eine","einer","eines","einem","einen","den","dem","im","in","an",
    "auf","zu","mit","für","von","vom","bei","aus","als","auch","doch","aber",
    "wenn","weil","wie","was","wer","wo","noch","nur","so","sehr","haben",
    "hat","hatte","hatten","ich","du","er","sie","es","wir","ihr","mein",
    "dein","sein","unser","kein","keine","ja","nein"
}

# Initialize tokenizer
enc = None
def get_tokenizer(model_name="text-embedding-3-large"):
    """Get or initialize the tokenizer for the specified model"""
    global enc
    if enc is None:
        # Fix for text-embedding-3-large model - use cl100k_base explicitly
        if model_name == "text-embedding-3-large":
            enc = tiktoken.get_encoding("cl100k_base")
        else:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                enc = tiktoken.get_encoding("cl100k_base")
    return enc

def tokenize_german(text: str) -> List[str]:
    """
    Lower-case, ASCII-folded, stop-word-filtered token split (German-optimised).
    Enhanced to handle financial document terms more effectively.
    """
    if not text:
        return []
    
    # Special handling for important financial terms (don't break these apart)
    important_terms = ["jahresrechnung", "jahresrechnungen", "jahresabrechnung", 
                      "jahresabrechnungen", "bilanz", "erfolgsrechnung"]
    
    # Make a copy of the text for term preservation
    text_lower = text.lower()
    
    # Preserve special terms by replacing them with unique tokens
    preserved_terms = {}
    for i, term in enumerate(important_terms):
        marker = f"__PRESERVED_TERM_{i}__"
        if term in text_lower:
            preserved_terms[marker] = term
            text_lower = text_lower.replace(term, marker)
    
    # Standard German text normalization
    text_lower = (text_lower
                .replace("ä", "ae").replace("ö", "oe")
                .replace("ü", "ue").replace("ß", "ss"))
    
    # Tokenize
    tokens = re.findall(r"\w+", text_lower)
    
    # Filter stopwords and restore preserved terms
    result = []
    for t in tokens:
        if t in preserved_terms:
            # Restore the original preserved term
            result.append(preserved_terms[t])
        elif t not in GERMAN_STOPWORDS:
            result.append(t)
    
    # Special handling for year mentions with text like "2023" or "2022"
    year_pattern = re.compile(r"20[12]\d")  # Matches years 2000-2029
    year_matches = year_pattern.findall(text)
    result.extend(year_matches)
    
    return result


# Alias for compatibility with new BM25 implementation
chunk_for_bm25 = tokenize_german


def truncate_to_tokens(txt: str, limit: int = 8000, model_name: str = "text-embedding-3-large") -> str:
    """
    Truncate text to a specified token limit
    
    Args:
        txt: The text to truncate
        limit: Maximum number of tokens
        model_name: The model to use for tokenization
        
    Returns:
        Truncated text
    """
    tokenizer = get_tokenizer(model_name)
    ids = tokenizer.encode(txt)
    return txt if len(ids) <= limit else tokenizer.decode(ids[:limit])

def batch_iter(texts: List[str], max_tokens: int = 16000, model_name: str = "text-embedding-3-large"):
    """
    Yield batches whose total tokens stay below the specified limit.
    
    Args:
        texts: List of texts to batch
        max_tokens: Maximum tokens per batch
        model_name: Model name for tokenization
        
    Yields:
        Batches of texts
    """
    tokenizer = get_tokenizer(model_name)
    batch, tok_sum = [], 0
    for t in texts:
        tk = len(tokenizer.encode(t))
        if batch and tok_sum + tk > max_tokens:
            yield batch
            batch, tok_sum = [], 0
        batch.append(t)
        tok_sum += tk
    if batch:
        yield batch

def normalize_title(title: str) -> str:
    """
    Normalize document titles to ensure consistent character representation
    between search results and actual files. This is critical for security
    to ensure document references match actual files in the system.
    
    Args:
        title: Document title to normalize
        
    Returns:
        Normalized title with consistent character representation
    """
    if not title:
        return ""
    
    # Replace special characters with a standard form
    # Most importantly, normalize slashes and underscores to ensure consistency
    normalized = title.replace(" / ", " _ ")
    normalized = normalized.replace("/", "_")
    
    # Handle document IDs in filenames (typically in curly braces)
    # Ensure they're preserved exactly as is
    id_pattern = r'__{.*?}__'
    id_matches = re.findall(id_pattern, normalized)
    
    # Temporarily replace IDs with placeholders to protect them during normalization
    for i, match in enumerate(id_matches):
        placeholder = f"__ID_PLACEHOLDER_{i}__"
        normalized = normalized.replace(match, placeholder)
    
    # Replace any multiple spaces with a single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove any trailing/leading whitespace
    normalized = normalized.strip()
    
    # Restore document IDs from placeholders
    for i, match in enumerate(id_matches):
        placeholder = f"__ID_PLACEHOLDER_{i}__"
        normalized = normalized.replace(placeholder, match)
    
    return normalized

def normalize_spacing_flexible(title: str) -> str:
    """
    More aggressive spacing normalization for flexible document matching.
    This function normalizes spacing around dashes and collapses multiple spaces.
    
    Args:
        title: Document title to normalize
        
    Returns:
        Title with normalized spacing for flexible matching
    """
    if not title:
        return ""
    
    # Remove file extension if present for matching
    if title.endswith('.pdf'):
        title = title[:-4]
    
    # Replace any multiple whitespace characters (spaces, tabs, etc.) with single space
    normalized = re.sub(r'\s+', ' ', title)
    
    # Normalize spacing around dashes - ensure single space before and after each dash
    # This handles cases like " - " vs " -  - " vs "- -" etc.
    normalized = re.sub(r'\s*-\s*', ' - ', normalized)
    
    # Collapse multiple consecutive dashes with spaces into a single dash
    # This handles "- - -" -> "-" and "- -" -> "-" patterns
    normalized = re.sub(r'(\s-\s)+', ' - ', normalized)
    
    # Clean up any resulting multiple spaces again
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove leading/trailing spaces
    normalized = normalized.strip()
    
    return normalized 