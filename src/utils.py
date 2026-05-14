# src/utils.py

import re
import unicodedata


def clean_english_text(text):
    """
    Basic cleaning for English text.
    We apply this to our current dataset.

    Steps:
    1. Lowercase — so "Joy" and "joy" are the same token
    2. Remove URLs — they carry no emotion signal
    3. Remove HTML tags — leftover from web scraping
    4. Remove punctuation — reduces noise
    5. Remove extra whitespace — clean up after removals
    """

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs
    # re.sub replaces all matches of a pattern with a string
    # r'http\S+' matches "http" followed by any non-whitespace characters
    text = re.sub(r'http\S+', '', text)

    # Step 3: Remove HTML tags like <br>, <p>, </div>
    text = re.sub(r'<.*?>', '', text)

    # Step 4: Remove punctuation — keep only letters and spaces
    # [^a-z\s] means "anything that is NOT a-z or whitespace"
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 5: Remove extra whitespace
    # \s+ matches one or more whitespace characters
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_bangla_text(text):
    """
    Cleaning pipeline specifically for Bangla social media text.

    This is more complex than English because:
    - Bangla Unicode has multiple representations of same character
    - Social media has heavy emoji usage
    - Code-switching (mixed Bangla-English) is common
    """

    # Step 1: Unicode normalization
    # NFC = Canonical Decomposition followed by Canonical Composition
    # This ensures হ and হ (same visual, different bytes) become identical
    text = unicodedata.normalize('NFC', text)

    # Step 2: Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Step 3: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Step 4: Remove emojis and special symbols
    # This regex matches characters outside standard Unicode text ranges
    text = re.sub(r'[^\u0000-\u017F\u0980-\u09FF\s]', '', text)
    # \u0980-\u09FF = Bangla Unicode block
    # \u0000-\u017F = Basic Latin + Latin Extended (for code-switching)

    # Step 5: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_text_stats(df, text_column='text'):
    """
    Returns basic statistics about a text column.
    Useful for before/after comparison of cleaning.
    """
    stats = {
        'total_samples': len(df),
        'empty_texts': df[text_column].apply(lambda x: len(str(x).strip()) == 0).sum(),
        'avg_word_count': df[text_column].apply(lambda x: len(str(x).split())).mean().round(2),
        'avg_char_count': df[text_column].apply(lambda x: len(str(x))).mean().round(2),
    }
    return stats
