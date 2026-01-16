word_to_num = {
    "không": "0",
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "năm": "5",
    "lăm": "5",
    "sáu": "6",
    "bảy": "7",
    "tám": "8",
    "chín": "9",
}

num_to_word = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

# Alternative pronunciations/words (used for variance in clean text)
num_to_word_alternatives = {
    "4": ["bốn", "tư"],
    # Add more alternatives if needed
}

# Corruption mappings for TTS (phonetic variations/mispronunciations)
word_corruptions = {
    "năm": ["năm", "lăm"],
    "chín": ["chín", "trín"],
    "bảy": ["bảy", "bẩy"],
    "tám": ["tám", "tớm"],
    "một": ["một", "mốt"],
    "bốn": ["bốn", "tư"],
    "lăm": ["lăm", "năm"],
}

# Mapping from all corrupted/alternative forms to standard forms
corrupted_to_standard = {
    "năm": "năm",
    "lăm": "năm",
    "chín": "chín",
    "trín": "chín",
    "bảy": "bảy",
    "bẩy": "bảy",
    "tám": "tám",
    "tớm": "tám",
    "một": "một",
    "mốt": "một",
    "bốn": "bốn",
    "tư": "bốn",
}


def normalize(text):
    """Normalize Vietnamese text by converting corrupted/alternative number words to standard forms.
    Also converts digits to their standard spoken form.
    
    Args:
        text: Input text with potentially corrupted number words or digits
        
    Returns:
        Normalized text with standard number words
    """
    text = str(text).lower().strip()
    
    # First, split text by spaces
    words = text.split()
    tokens = []
    
    # For each word, split digits and non-digits into separate tokens
    for word in words:
        current_token = ""
        prev_is_digit = None
        
        for c in word:
            is_digit = c.isdigit()
            
            # If character type changes or it's a digit, start new token
            if prev_is_digit is not None and (is_digit != prev_is_digit or is_digit):
                if current_token:
                    tokens.append(current_token)
                current_token = c
            else:
                current_token += c
            
            prev_is_digit = is_digit
        
        if current_token:
            tokens.append(current_token)
    
    # Now normalize each token
    normalized_words = []
    for token in tokens:
        # Check if token is in corrupted mapping
        if token in corrupted_to_standard:
            normalized_words.append(corrupted_to_standard[token])
        # Check if token is a single digit
        elif token in num_to_word:
            normalized_words.append(num_to_word[token])
        else:
            normalized_words.append(token)
    
    result = " ".join(normalized_words)
    return result if result else ""
if __name__=="__main__":
    print(normalize("01234"))
    print(normalize("01234năm"))
    print(normalize("01234năm 67tám9"))