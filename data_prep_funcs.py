import math
import os
import pprint
from pprint import pformat, pprint
import json
import random
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Tuple
import abbrevs
from finetune_sys_prompt import finetune_sys_prompt
from anki_helper_classes import _ClozeType, _NoteType, AnkiRow, TopicCloze, ExCloze
from data_helper_classes import ClassToDictEncoder, MsgExchange, MsgObj, Conversation, RRJRDataset, RRJRDatasetDict
from rand_var import RandVar
import string
# custom_dir = 'D:/DocumentsHDD/Coding_Scrap/python/ollama/clozer/nltk_data'  # Change this to your preferred path
custom_dir = '.venv\\nltk_data'  # Change this to your preferred path
os.environ['NLTK_DATA'] = custom_dir
empty_resp = []
# empty_resp = {"anki_notes": [], "sources":{}}

def g_empty_exchange(system_prompt = None) -> MsgExchange:
    user = MsgObj("user", "")
    assistant = MsgObj("assistant", empty_resp)
    return MsgExchange(user, assistant, system_prompt if system_prompt else None)
def g_empty_convo(system_prompt = None) -> Conversation:
    return Conversation(g_empty_exchange(system_prompt))


unicode_set_args = (None, "nums", "roman", "letters", "special", "whitespace")
import random
import string

def random_unicode_string(max_length=100, char_set=None):
    """
    Generate a random string of Unicode characters broken into chunks
    based on typical English word length distribution.
    
    Args:
        max_length (int): Maximum length of the string to generate
        char_set (str): Type of characters to include:
            - None (default): No limit, any valid UTF-8 Unicode character
            - "roman": Only Roman alphabet letters (a-z, A-Z)
            - "nums": Only numeric characters (0-9)
            - "letters": Letters from any alphabet (valid UTF-8 only)
            - "special": Only special characters
            - "whitespace": Only whitespace characters
    
    Returns:
        str: A space-separated string of random Unicode chunks
    """
    # Approximate distribution of English word lengths
    # Based on common distribution patterns
    word_length_distribution = {
        1: 0.03,  # 3% of words are 1 character
        2: 0.08,  # 8% of words are 2 characters
        3: 0.17,  # 17% of words are 3 characters
        4: 0.21,  # 21% of words are 4 characters
        5: 0.17,  # 17% of words are 5 characters
        6: 0.12,  # 12% of words are 6 characters
        7: 0.08,  # 8% of words are 7 characters
        8: 0.05,  # 5% of words are 8 characters
        9: 0.04,  # 4% of words are 9 characters
        10: 0.03,  # 3% of words are 10 characters
        11: 0.02,  # 2% of words are 11+ characters
    }
    
    # Valid Unicode ranges for UTF-8 (excluding surrogates and invalid code points)
    valid_ranges = [
        (0x0000, 0xD7FF),    # Basic Multilingual Plane (before surrogates)
        (0xE000, 0xFFFF),    # BMP after surrogates
        (0x10000, 0x10FFFF)  # Supplementary planes
    ]
    
    # Select characters based on char_set parameter
    if char_set == "roman":
        # Only Roman alphabet letters (a-z, A-Z)
        char_pool = string.ascii_letters
        random_chars = [random.choice(char_pool) for _ in range(max_length)]
    
    elif char_set == "nums":
        # Only numeric characters (0-9)
        char_pool = string.digits
        random_chars = [random.choice(char_pool) for _ in range(max_length)]
    
    elif char_set == "letters":
        # Letters from any alphabet (valid UTF-8 only)
        random_chars = []
        for _ in range(max_length):
            # Generate a code point and check if it's a letter
            while True:
                # Randomly select a range first, then a code point within that range
                start, end = random.choice(valid_ranges)
                code_point = random.randint(start, end)
                    
                char = chr(code_point)
                # Check if the character is a letter in any script
                if char.isalpha():
                    random_chars.append(char)
                    break
    
    elif char_set == "whitespace":
        # Only whitespace characters
        whitespace_chars = [
            ' ',        # Space
            '\t',       # Tab
            '\n',       # Newline
            '\r',       # Carriage return
            '\f',       # Form feed
            '\v',       # Vertical tab
            '\u00A0',   # Non-breaking space
            '\u2000',   # En quad
            '\u2001',   # Em quad
            '\u2002',   # En space
            '\u2003',   # Em space
            '\u2004',   # Three-per-em space
            '\u2005',   # Four-per-em space
            '\u2006',   # Six-per-em space
            '\u2007',   # Figure space
            '\u2008',   # Punctuation space
            '\u2009',   # Thin space
            '\u200A',   # Hair space
            '\u200B',   # Zero-width space
            '\u2028',   # Line separator
            '\u2029',   # Paragraph separator
            '\u202F',   # Narrow no-break space
            '\u205F',   # Medium mathematical space
            '\u3000',   # Ideographic space
        ]
        random_chars = [random.choice(whitespace_chars) for _ in range(max_length)]
    
    elif char_set == "special":
        # Special characters (punctuation, symbols, etc.)
        special_ranges = [
            (0x0021, 0x002F),  # Basic Latin punctuation 1
            (0x003A, 0x0040),  # Basic Latin punctuation 2
            (0x005B, 0x0060),  # Basic Latin punctuation 3
            (0x007B, 0x007E),  # Basic Latin punctuation 4
            (0x00A0, 0x00BF),  # Latin-1 punctuation and symbols
            (0x2000, 0x206F),  # General punctuation
            (0x2070, 0x209F),  # Superscripts and subscripts
            (0x20A0, 0x20CF),  # Currency symbols
            (0x2100, 0x214F),  # Letterlike symbols
            (0x2190, 0x21FF),  # Arrows
            (0x2200, 0x22FF),  # Mathematical operators
            (0x2300, 0x23FF),  # Miscellaneous technical
            (0x2400, 0x243F),  # Control pictures
            (0x2440, 0x245F),  # Optical character recognition
            (0x2460, 0x24FF),  # Enclosed alphanumerics
            (0x2500, 0x257F),  # Box drawing
            (0x2580, 0x259F),  # Block elements
            (0x25A0, 0x25FF),  # Geometric shapes
            (0x2600, 0x26FF),  # Miscellaneous symbols
            (0x2700, 0x27BF),  # Dingbats
        ]
        
        # Flatten the special character ranges - all these are already in valid UTF-8 range
        special_chars = []
        for start, end in special_ranges:
            special_chars.extend(chr(code) for code in range(start, end + 1))
        
        random_chars = [random.choice(special_chars) for _ in range(max_length)]
    
    else:  # None or default case
        # Any valid UTF-8 Unicode character
        random_chars = []
        for _ in range(max_length):
            # Select a valid Unicode range first, then a code point within that range
            start, end = random.choice(valid_ranges)
            code_point = random.randint(max(0x0020, start), end)  # Start from at least space character (0x0020)
            random_chars.append(chr(code_point))
    
    random_string = ''.join(random_chars)
    
    # Now break it into chunks according to the distribution
    chunks = []
    current_pos = 0
    
    while current_pos < len(random_string):
        # Choose a random word length based on the distribution
        length_choices = list(word_length_distribution.keys())
        length_weights = list(word_length_distribution.values())
        chunk_length = random.choices(length_choices, weights=length_weights, k=1)[0]
        
        # Make sure we don't go beyond the string length
        end_pos = min(current_pos + chunk_length, len(random_string))
        
        # Add the chunk to our list
        chunk = random_string[current_pos:end_pos]
        
        # Apply capitalization rules for letter-based character sets
        if char_set in ["roman", "letters"] and chunk:
            # 5% chance for the whole word to be capitalized
            if random.random() < 0.02:
                chunk = chunk.upper()
            # 5% chance for the first letter to be capitalized (title case)
            elif random.random() < 0.01:
                if len(chunk) > 0:
                    chunk = chunk[0].upper() + chunk[1:]
            # 50% chance for the first chunk to start with a capital letter
            elif len(chunks) == 0 and random.random() < 0.5:
                if len(chunk) > 0:
                    chunk = chunk[0].upper() + chunk[1:]
            # Otherwise, capitalize according to English text ratio (about 5% of letters)
            else:
                new_chunk = ""
                for char in chunk:
                    if char.isalpha() and random.random() < 0.005:
                        new_chunk += char.upper()
                    else:
                        new_chunk += char
                chunk = new_chunk
        
        chunks.append(chunk)
        
        # Move to the next position
        current_pos = end_pos
    
    # Join the chunks with spaces
    return ' '.join(chunks)

class InvalidSpecialTag(Exception):
    """Exception raised when an invalid special tag format is provided."""
    pass
def is_valid_special_tag(tag_html):
    """
    Validates if a string is a proper HTML opening tag.
    
    Args:
        tag_html (str): The HTML tag to validate
        
    Returns:
        bool: True if the tag is valid, False otherwise
    """
    return bool(re.match(r'<[a-z][a-z0-9]*(\s+[a-z][a-z0-9-]*="[^"]*")*\s*>', tag_html, re.IGNORECASE))
def create_html_list(words, list_type):
    """
    Create an HTML ordered or unordered list from a list of words.
    
    Args:
        words (list): List of words to be turned into list items
        list_type (str): Either 'ul' or 'ol'
        
    Returns:
        str: HTML list with each word as a list item
    """
    if not words:
        return ""
        
    result = [f"<{list_type}>"]
    for word in words:
        result.append(f"<li>{word}</li>")
    result.append(f"</{list_type}>")
    
    return " ".join(result)

def random_html_tagger(input_string, max_depth=3, tag_density=0.3, br_density=0.1, empty_tag_chance=0.01, 
                    bias=None, wrap_whole=0.5, regular_tags=None, special_tags=None, container_tags=None):
    """
    Randomly wraps parts of a string in HTML tags with support for nested structures.
    
    Args:
        input_string (str): The string to be wrapped with HTML tags
        max_depth (int): Maximum nesting level for tags
        tag_density (float): Probability of applying tags to chunks (0.0 to 1.0)
        br_density (float): Probability of inserting <br> tags between words (0.0 to 1.0)
        empty_tag_chance (float): Probability of inserting empty tags (0.0 to 1.0)
        bias (dict): Dictionary mapping tag names to bias probabilities (0.0 to 1.0)
               e.g., {'span_code': 0.3} gives a 30% chance to override with <span class="code">
        wrap_whole (float): Probability of wrapping the entire output in a container tag (0.0 to 1.0)
        regular_tags (list): Override the default regular tags list
        special_tags (dict): Dictionary of special tags in format {'keyword': '<tag attr="value">'}
        container_tags (list): Override the default container tags list
    
    Returns:
        str: The input string randomly tagged with HTML elements
        
    Raises:
        InvalidSpecialTag: If a special tag does not match valid HTML tag format
    """
    # Set default tags if not provided
    if regular_tags is None:
        regular_tags = ['div', 'span', 'pre', 'strong']
    
    # List tags remain constant
    list_tags = ['ul', 'ol']
    
    # Process special tags if provided
    processed_special_tags = {}
    special_tag_keys = []
    
    if special_tags:
        for key, tag_html in special_tags.items():
            # Validate that the tag is a proper HTML opening tag
            if not is_valid_special_tag(tag_html):
                raise InvalidSpecialTag(f"Invalid special tag format: {tag_html}")
            
            # Extract the tag name for proper closing
            tag_name = extract_tag_name(tag_html)
            processed_special_tags[key] = {
                'open': tag_html,
                'close': f'</{tag_name}>'
            }
            special_tag_keys.append(key)
    
    # Combine all tag types
    all_tags = regular_tags + list_tags + special_tag_keys
    
    # Set default container tags if not provided
    if container_tags is None:
        container_tags = ['div', 'span', 'pre']
    
    # Add special tags to container tags if applicable
    container_tags = list(container_tags)  # Make a copy to avoid modifying the original
    if special_tags:
        container_tags.extend(special_tag_keys)
    
    # Initialize bias dictionary if None
    if bias is None:
        bias = {}
        
    # Normalize bias values to be between 0 and 1
    for tag in bias:
        bias[tag] = max(0, min(1, bias[tag]))
    
    # Split the string into words
    words = input_string.split()
    
    def tag_substring(word_list, depth=0):
        if not word_list or depth >= max_depth:
            return ' '.join(word_list)
        
        result = []
        i = 0
        
        while i < len(word_list):
            # Randomly insert empty tags
            if random.random() < empty_tag_chance:
                tag = random.choice(regular_tags)  # Only use regular tags for empty tags
                
                # Apply bias for empty tags too
                for biased_tag, bias_value in bias.items():
                    if random.random() < bias_value and biased_tag in regular_tags:
                        tag = biased_tag
                        break
                result.append(f"<{tag}></{tag}>")
            
            # Decide if we want to apply a tag
            if random.random() < tag_density and i < len(word_list) - 1:
                # Choose a random tag
                tag = random.choice(all_tags)
                
                # Apply bias - chance to override the chosen tag with a biased option
                for biased_tag, bias_value in bias.items():
                    if random.random() < bias_value:
                        tag = biased_tag
                        break
                
                # Determine how many words to include in this tag (1 to 5)
                words_to_include = min(random.randint(1, 5), len(word_list) - i)
                chunk = word_list[i:i+words_to_include]
                
                # Handle lists and special tags differently
                if tag in list_tags:
                    # Use the dedicated list creation function
                    list_html = create_html_list(chunk, tag)
                    result.append(list_html)
                elif tag in special_tag_keys:
                    # Handle custom special tags
                    inner_content = tag_substring(chunk, depth + 1)
                    result.append(f"{processed_special_tags[tag]['open']}{inner_content}{processed_special_tags[tag]['close']}")
                else:
                    # Process this chunk recursively with increased depth
                    inner_content = tag_substring(chunk, depth + 1)
                    
                    # Add this tagged section
                    result.append(f"<{tag}>{inner_content}</{tag}>")
                
                i += words_to_include
            else:
                # Add word without any tag
                result.append(word_list[i])
                i += 1
                
                # Randomly insert <br> after a word
                if i < len(word_list) and random.random() < br_density:
                    result.append("<br>")
        
        return ' '.join(result)
    
    result = tag_substring(words)
    
    # 50/50 chance (or specified probability) to wrap the entire result in a container tag
    if random.random() < wrap_whole:
        # Choose a random container tag
        container_tag = random.choice(container_tags)
        
        # Apply bias to container tag selection if applicable
        for biased_tag, bias_value in bias.items():
            if random.random() < bias_value and biased_tag in container_tags:
                container_tag = biased_tag
                break
        
        # Wrap the entire content
        if container_tag in special_tag_keys:
            result = f"{processed_special_tags[container_tag]['open']}{result}{processed_special_tags[container_tag]['close']}"
        else:
            result = f"<{container_tag}>{result}</{container_tag}>"
    
    return result

def is_valid_special_tag(tag_html):
    """
    Validates if a string is a proper HTML opening tag.
    
    Args:
        tag_html (str): The HTML tag to validate
        
    Returns:
        bool: True if the tag is valid, False otherwise
    """
    return bool(re.match(r'<[a-z][a-z0-9]*(\s+[a-z][a-z0-9-]*="[^"]*")*\s*>', tag_html, re.IGNORECASE))

def extract_tag_name(tag_html):
    """
    Extracts the tag name from an HTML opening tag.
    
    Args:
        tag_html (str): The HTML tag
        
    Returns:
        str: The tag name
    """
    match = re.match(r'<([a-z][a-z0-9]*)', tag_html, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def random_html_tags(count=10, valid_html=False, regular_tags=None, special_tags=None):
    """
    Generates random HTML tags.
    
    Args:
        count (int): Number of tags (or tag pairs if valid_html is True)
        valid_html (bool): If True, generates valid HTML with matching open/close tags
        regular_tags (list): List of regular HTML tags to use
        special_tags (dict): Dictionary of special tags in format {'keyword': '<tag attr="value">'}
    
    Returns:
        str: A string containing random HTML tags
        
    Raises:
        InvalidSpecialTag: If a special tag does not match valid HTML tag format
    """
    # Set default tags if not provided
    if regular_tags is None:
        regular_tags = ['div', 'span', 'pre']
    
    # Process special tags if provided
    processed_special_tags = {}
    special_tag_keys = []
    
    if special_tags:
        for key, tag_html in special_tags.items():
            # Validate that the tag is a proper HTML opening tag
            if not is_valid_special_tag(tag_html):
                raise InvalidSpecialTag(f"Invalid special tag format: {tag_html}")
            
            # Extract the tag name for proper closing
            tag_name = extract_tag_name(tag_html)
            if not tag_name:
                raise InvalidSpecialTag(f"Could not extract tag name from: {tag_html}")
                
            processed_special_tags[key] = {
                'open': tag_html,
                'close': f'</{tag_name}>'
            }
            special_tag_keys.append(key)
    
    # Combine all tag types
    all_tags = regular_tags + special_tag_keys
    
    if valid_html:
        # Generate valid HTML with matching opening and closing tags
        result = []
        open_tags = []
        
        # Generate opening tags
        for _ in range(count):
            tag = random.choice(all_tags)
            
            if tag in special_tag_keys:
                result.append(processed_special_tags[tag]['open'])
                open_tags.append(processed_special_tags[tag]['close'])
            else:
                result.append(f"<{tag}>")
                open_tags.append(f"</{tag}>")
        
        # Add closing tags in reverse order
        while open_tags:
            result.append(open_tags.pop())
            
        return ''.join(result)
    else:
        # Generate random tags without ensuring valid HTML
        result = []
        
        for _ in range(count):
            tag = random.choice(all_tags)
            is_opening = random.choice([True, False])
            
            if tag in special_tag_keys:
                if is_opening:
                    result.append(processed_special_tags[tag]['open'])
                else:
                    result.append(processed_special_tags[tag]['close'])
            else:
                if is_opening:
                    result.append(f"<{tag}>")
                else:
                    result.append(f"</{tag}>")
        
        return ''.join(result)


def g_random_words(count=1, min_length=None, max_length=None):
    from nltk.corpus import words

    """
    Returns a string of random words from the NLTK words corpus.
    
    Args:
        count (int): Number of random words to select
        min_length (int): Minimum length of words to include
        max_length (int): Maximum length of words to include (None for no maximum)
    
    Returns:
        str: Space-separated string of random words
    """
    # Get the word list from NLTK
    word_list = words.words()
    
    # Filter words by length if needed
    filtered_words = [word for word in word_list 
                     if (not min_length or len(word) >= min_length) 
                     and (not max_length or len(word) <= max_length)]
    
    # Select random words
    selected_words = []
    for _ in range(count):
        selected_words.append(random.choice(filtered_words))
    
    # Join words into a string
    return ' '.join(selected_words)
# print(
#     random_html_tagger(
#         random_unicode_string(char_set="roman"),
#         container_tags={"span", "div", "pre", "span_code"},
#         special_tags={"span_code": "<span class=\"code\">"},
#         bias={"span_code": 0.1},
#         wrap_whole=1.0
#     )
# )
# print(random_html_tags(valid_html=False, special_tags={"span_close": "<span class=\"code\">"}))
def g_adjusted_vaguery():
    from vagueries.vagueries import vagueries
    v = random.choice(vagueries)
    if random.random() < 0.5:
        v.replace(",", "")
    v = abbrevs.abbrev(v, pref = "random", chance = 0.7)
    if random.random() < 0.5:
        v = v[0].upper() + v[1:]
    else:
        v = v[0].lower() + v[1:]
    return v
            
def g_random_garbage():
    choices = [g_random_words, random_unicode_string, random_html_tags]
    out_str = ""
    choice = random.choice(choices)
    if choice == g_random_words:
        word_count = random.randint(5, 37)
        out_str = g_random_words(word_count)
    elif choice == random_unicode_string:
        max_length = random.randint(25, 404)
        out_str =  random_unicode_string(max_length, random.choice(unicode_set_args))
    elif choice == random_html_tags:
        max_count = random.randint(5, 13)
        if bool(random.randint(0,1)):
            out_str = random_html_tags(max_count, valid_html=True)
        else:
            out_str = random_html_tags(max_count * 2, valid_html=False)
    # tag up or not randomly
    
    max_length = random.randint(25, 404)
    random_html_tagger_args = {
        "regular_tags": ['div', 'span', 'pre', 'b', 'strong', 'em'], # these will be used by sub-strs
        "container_tags": ["span", "div", "pre", "span_code"], # these could wrap the whole return str
        "special_tags": {"span_code": "<span class=\"code\">"}, # extra tags to add
        "bias": {"span_code": 0.05}, # making the special tag slightly more likely to appear
    }
    if random.random() <= 0.5:
        out_str = random_html_tagger(out_str, **random_html_tagger_args)
    return out_str

def mk_screwed_up_convos(system_msg: MsgObj = None) -> list[Conversation]:
    """returns a list of conversations with werid inputs that should be ignored during testing"""
    from vagueries.vagueries import vagueries
    to_return = []
    # just random html both valid and invalid
    rand_html_min, rand_html_max = 15, 30
    html_count_invalid = random.randint(rand_html_min, rand_html_max)
    html_count = RandVar(15, 30)
    master_chance = 0.7
    # master_chance = 0.0
    # if True: # Only invalid html
    invalid_ignore_master = False
    if random.random() <= master_chance if not invalid_ignore_master else 0.0: # Only invalid html
        to_return.append(Conversation(
            MsgExchange(
                MsgObj("user", random_html_tags(count = html_count.roll(), valid_html=False)),
                MsgObj("assistant", empty_resp),
                system_msg
            )
        ))
    html_count_valid = random.randint(rand_html_min, rand_html_max)
    generate_w_valid_html_ignore_master = False
    if random.random() <= master_chance if not invalid_ignore_master else 0.0: # generate with valid html
        to_return.append(Conversation(
            MsgExchange(
                MsgObj("user", random_html_tags(count = html_count.roll() // 2, valid_html=True)),
                MsgObj("assistant", empty_resp),
                system_msg
            )
        ))
    
    # random strings. various random unicode strings.
    unicode_str_args = (None, "roman", "nums", "special", "whitespace") # Modify as needed
    # unicode_str_args = ((),) # or uncomment to disable <---------------------
    random_html_tagger_args = {
        "container_tags": {"span", "div", "pre", "span_code"},
        "special_tags": {"span_code": "<span class=\"code\">"},
        "bias": {"span_code": 0.05},
    }
    unicode_str_length = RandVar(16, 133)
    random_str_ignore_master = False
    if random.random() <= master_chance if not random_str_ignore_master else 0.0:
        for a in unicode_str_args:
            for step in range(1, 3):
                res_str = None
                if step == 1:# just the string itself
                    res_str = random_unicode_string(unicode_str_length.roll(), a)
                elif step == 2: # with random tags
                    res_str = random_html_tagger(
                        random_unicode_string(unicode_str_length.roll(), a),
                        **random_html_tagger_args
                    )
                to_return.append(Conversation(MsgExchange(
                    MsgObj("user", res_str),
                    MsgObj("assistant", empty_resp),
                    system_msg
                )))
    # similar thing but with jsut random words
    count_rand_words_min, count_rand_words_max = 8, 77
    count_rand_words = RandVar(8, 77)
    random_words_ignore_master = False
    if random.random() <= master_chance if not random_words_ignore_master else 0.0:
        res_str = g_random_words(count_rand_words.roll())
        to_return.append(Conversation(MsgExchange(
            MsgObj("user", res_str),
            MsgObj("assistant", empty_resp),
            system_msg
        )))
    r_words_w_tags_ignore_master = False
    if random.random() <= master_chance if not r_words_w_tags_ignore_master else 0.0: # with random tags
        res_str = random_html_tagger(
            g_random_words(count_rand_words.roll()),
            **random_html_tagger_args
        )
        to_return.append(Conversation(MsgExchange(
            MsgObj("user", res_str),
            MsgObj("assistant", empty_resp),
            system_msg
        )))
    empty_convo_ignore_master = False
    if random.random() <= master_chance if not empty_convo_ignore_master else 0.0: # the empty convo
        to_return.append(g_empty_convo())
    adj_vaguery_ignore_master = False
    if random.random() <= master_chance if not adj_vaguery_ignore_master else 0.0:
        to_return.append(Conversation(MsgExchange(
            MsgObj("user", g_adjusted_vaguery()),
            MsgObj("assistant", empty_resp),
            system_msg
        )))
    # mult-block inputs
    # options
    
    mult_block_file_count = RandVar(1, 4, val = 3)
    mult_block_count = RandVar(1, 10, val=5)
    str_gen_choices: List[Tuple[Callable, Tuple, Dict]] = [
        (random_html_tags
            ,(html_count.roll(),)
            ,{"valid_html": False})
        ,(random_html_tags
            ,(html_count.roll() // 2,)
            ,{"valid_html": True})
        ,(g_random_words, (count_rand_words.roll(),))
        ,(random_html_tagger, (g_random_words(count_rand_words.roll()),), random_html_tagger_args)
        ,(lambda: "",)
        ,(g_adjusted_vaguery,)
    ]
    if True and len(unicode_set_args) > 0:
        str_gen_choices.extend((
            (random_unicode_string
                ,(unicode_str_length.roll(), random.choice(unicode_str_args)))
            ,(random_html_tagger
                ,(random_unicode_string(unicode_str_length.roll(), random.choice(unicode_str_args)),)
                ,random_html_tagger_args)
        ))
    # print(pformat(tuple(type(e) for e in str_gen_choices)))
    def _call(_choice: Tuple[Callable, Tuple, Dict]):
        try:
            if len(_choice) == 3:
                return _choice[0](*_choice[1], **_choice[2])
            elif len(_choice) == 2:
                return _choice[0](*_choice[1])
            elif len(_choice) == 1:
                return _choice[0]()
            else:
                raise Exception(f"_choice has too many elements {len(_choice)}\n  {_choice}")
        except Exception as e:
            raise Exception(f"error occured trying to call _choice\n  {_choice}")
    multi_block_ignore_master = False
    if random.random() <= master_chance if not multi_block_ignore_master else 0.0:
        # mult_block_file_count.val = 3
        mult_block_file_count.roll()
        for i in range(mult_block_file_count.val):
            # print(i)
            # mult_block_count.val = 4
            mult_block_count.roll()
            ffile_str = ""
            first = True
            for j in range(mult_block_count.val):
                # add separators. 30% of the time at the beginning of the file as well
                if not first or random.random() < 0.3:
                    if random.random() <= 0.5:
                        ffile_str += f"\n{g_random_sparator() * random.randint(1,2)}\n"
                    else:
                        ffile_str += "\n" + g_random_sparator() * random.randint(2, 14) + "\n"
                else: first = False

                ffile_str += _call(random.choice(str_gen_choices))
            if random.random() <= 0.3:
                if random.random() <= 0.5:
                    ffile_str += f"\n{g_random_sparator() * random.randint(1,2)}"
                else:
                    ffile_str += "\n" + g_random_sparator() * random.randint(2, 14)
                if random.random() <= 0.5:  # Optionally add newline at the end
                    ffile_str += f"\n" * random.randint(1,2)
            to_return.append(Conversation(MsgExchange(
                MsgObj("user", ffile_str),
                MsgObj("assistant", empty_resp),
                system_msg
            )))
    print("mk_screwed_up_convos to_return len", len(to_return))
    return to_return

def g_random_sparator():
    vals = ('-', '–', '=', "_")
    return random.choice(vals)
def g_screwed_up_convos_old(system_msg: MsgObj = None) -> list[Conversation]:
    """returns a list of conversations with werid inputs that should be ignored during testing"""
    from vagueries.vagueries import vagueries
    to_return = []
    # just random html both valid and invalid
    len_rand_hmtl = random.randint()
    ran_min, ran_max = 15, 30
    html_count_invalid = random.randint(ran_min, ran_max)
    html_count_valid = random.randint(ran_min, ran_max)
    to_return.append(Conversation(
        MsgExchange(
            MsgObj("user", random_html_tags(count = html_count_invalid, valid_html=False)),
            MsgObj("assistant", empty_resp),
            system_msg
        )
    ))
    to_return.append(Conversation(
        MsgExchange(
            MsgObj("user", random_html_tags(count = html_count_valid // 2, valid_html=True)),
            MsgObj("assistant", empty_resp),
            system_msg
        )
    ))

    # random strings. various random unicode strings.
    unicode_str_args = (None, "roman", "letters", "nums", "special")
    random_html_tagger_args = {
        "container_tags": {"span", "div", "pre", "span_code"},
        "special_tags": {"span_code": "<span class=\"code\">"},
        "bias": {"span_code": 0.05},
    }
    for a in unicode_str_args:
        for step in range(1, 3):
            str_length = random.randint(16, 333)
            res_str = None
            if step == 1:# just the string itself
                res_str = random_unicode_string(str_length, a)
            elif step == 2: # with random tags
                res_str = random_html_tagger(
                    random_unicode_string(str_length, a),
                    **random_html_tagger_args
                )
            to_return.append(Conversation(MsgExchange(
                MsgObj("user", res_str),
                MsgObj("assistant", empty_resp),
                system_msg
            )))
    # similar thing but with jsut random words
    count_rand_words = random.randint(8, 33)
    for a in unicode_str_args:
        for step in range(1, 3):
            str_length = random.randint(16, 242)
            if step == 1: # just the string itself
                res_str = g_random_words(count_rand_words)
            elif step == 2: # with random tags
                res_str = random_html_tagger(
                    g_random_words(count_rand_words),
                    **random_html_tagger_args
                )
            to_return.append(Conversation(MsgExchange(
                MsgObj("user", res_str),
                MsgObj("assistant", empty_resp),
                system_msg
            )))
    # the empty convo
    to_return.append(g_empty_convo())
    return to_return
    
def mid_plus_minus_k(data, k):
    data = sorted(data)
    print(data, f"k == {k}")
    n = len(data)
    # print ("n == ", n)
    isOdd = bool(n % 2)
    # print("isOdd", isOdd)
    middle = (n // 2) 
    # -1 to make indexif odd needs to be +1 which cancles out
    if not isOdd: middle -= 1 
    # print("middle", middle)
    start = middle - k if isOdd else middle - (k - 1)
    end = middle + k
    # print(start, middle, end)
    to_return = data[start: end+1]
    print(to_return)
    return to_return
# print(g_random_words(10, max_length=9))
# print(random_html_tagger(""))
if __name__ == "__main__":
    
    print(mid_plus_minus_k((x for x in range(10)), 1))
    print(mid_plus_minus_k((x for x in range(11)), 1))
    print(mid_plus_minus_k((x for x in range(10)), 2))
    print(mid_plus_minus_k((x for x in range(13)), 2))
    print(mid_plus_minus_k((x for x in range(13)), 1))
    # from finetune_sys_prompt import finetune_sys_prompt
    # convos = mk_screwed_up_convos({"role": "system", "content": finetune_sys_prompt})
    # for i in range(len(convos) - 1, -1, -1):
    #     # print("test", end='')
    #     print(convos[i].exchanges[0].user.content)
    #     input(f"content from i = {i}")
    #     print("\n"+"88888_ignore_this88888_this_is_a_sparator_88888"* 2 + "\n")