import re
import random
from typing import List, Tuple, Union, Set, Collection
from enum import Enum

class CapsRule(Enum):
    NoFirst = 0
    NoAll = 1

class Abbr:
    """a single abbreviation with its CapsRules"""
    
    def __init__(self, text_or_tuple: Union[str, Tuple], caps_rules: Collection[CapsRule] = None):
        """
        Initialize an Abbr object.
        
        Args:
            text_or_tuple: Either:
                - A string (the abbreviation text)
                - A tuple with first element as string and remaining elements as CapsRules
            caps_rules: Set of capitalization rules to apply (used only when text_or_tuple is a string)
        """
        if isinstance(text_or_tuple, tuple):
            if len(text_or_tuple) == 0:
                raise ValueError("Tuple must not be empty")
            
            self.text = str(text_or_tuple[0])
            self.caps_rules = set()
            # Remaining elements should be CapsRules
            for i in range(1, len(text_or_tuple)):
                rule = text_or_tuple[i]
                if not isinstance(text_or_tuple[i], CapsRule):
                    raise ValueError(f"Every value after first in text_or_tuple should be a CapsRule found {type(text_or_tuple[i]).__name__}")
                self.caps_rules.add(rule)
        else:
            self.text = str(text_or_tuple)
            self.caps_rules = set(caps_rules) if caps_rules else set()
    
    def __repr__(self):
        return f"Abbr('{self.text}', {self.caps_rules})"

class AbbrsCol:
    """
    Represents a collection of a preferred abbreviation, its variations and patterns that should match to it.
   
    Attributes:
        patterns (tuple): Original text patterns to match for abbreviation
        pattern_to_comp (dict): Dictionary mapping original patterns to compiled regex patterns
        pref (Abbr): The preferred abbreviation
        not_pref (tuple): Additional variations of the abbreviation (excludes preferred)
    """
   
    def __init__(self, patterns: Tuple[str, ...], 
                pref: Union[Abbr, Tuple[str, Set[CapsRule]], str], 
                not_pref: Union[Tuple[Union[Tuple, Abbr], ...], None]=None):
        """
        Initialize an AbbrsCol object.
       
        Args:
            patterns (tuple): Original text patterns to match
            pref: The preferred abbreviation, either as:
                - Abbr object
                - Tuple of (str, Set[CapsRule])
                - str (no capitalization rules)
            not_pref (tuple, optional): Additional variations of the abbreviation (excludes preferred)
                Each element must be either an Abbr object or a Tuple
        """
        self.patterns = patterns
        def _get_abbr(inp) -> Abbr:
            if isinstance(inp, Abbr):
                return inp
            if isinstance(inp, tuple):
                return Abbr(inp)
            raise TypeError(f"inp was not Abbr or tuple, got {type(item).__name__}")

        # Convert pref to Abbr if it's not already
        self.pref = _get_abbr(pref)
       
        # Convert not_pref elements to Abbr objects
        if not_pref is None:
            self.not_pref = ()
        else:
            # Process each element in not_pref, converting to Abbr if needed
            processed_not_pref = []
            for item in not_pref:
                # print(self.pref.text, "sending ", item)
                abbr_item = _get_abbr(item)                
                
                # print(self.pref.text, "adding ", abbr_item)
                if abbr_item.text != self.pref.text:
                    processed_not_pref.append(abbr_item)
            
            self.not_pref = tuple(processed_not_pref)
       
        # Compile regex patterns
        self.pattern_to_comp = {}
        for pattern in patterns:
            self.pattern_to_comp[pattern] = re.compile(pattern, re.IGNORECASE)
   
    def all(self):
        """
        Returns all abbreviations (preferred and variations) as a tuple.
       
        Returns:
            tuple: All available abbreviations with preferred first
        """
        return (self.pref,) + self.not_pref

# Create the list of abbreviations with condensed initialization
abbrs: List[AbbrsCol] = [
    # Javascript
    AbbrsCol(
        ("javascript", "java script"),
        Abbr("js", caps_rules = (CapsRule.NoFirst,)),
        (Abbr("JS", caps_rules = None),)
    ),
    
    # Python
    AbbrsCol(
        ("python",), 
        Abbr("py"),
    ),
    
    # Various abbreviations with "/"
    AbbrsCol(
        ("something",), 
        Abbr("s/th"),
        (Abbr("s/t"),)
    ),
    AbbrsCol(
        ("somewhere",),
        Abbr("s/wr"),
        (Abbr("s/w"),)
    ),
    AbbrsCol(
        ("someone",),
        Abbr("s/o"),
    ),
    AbbrsCol(
        ("somebody",),
        Abbr("s/b"),
    ),
    AbbrsCol(
        ("with",),
        Abbr("w/"),
    ),
    
    # Grammatical cases
    AbbrsCol(
        ("nominative",),
        Abbr("nom"),
    ),
    AbbrsCol(
        ("accusative",),
        Abbr("acc"),
    ),
    AbbrsCol(
        ("dative",),
        Abbr("dat"),
    ),
    AbbrsCol(
        ("genitive",),
        Abbr("gen"),
    ),
    AbbrsCol(
        ("locative",),
        Abbr("loc"),
    ),
    AbbrsCol(
        ("instrumental",),
        Abbr("ins"),
    ),
    AbbrsCol(
        ("comitative",),
        Abbr("com"),
    ),
    AbbrsCol(
        ("vocative",),
        Abbr("voc"),
    ),
    
    # Verb forms
    AbbrsCol(
        ("infinitive",),
        Abbr("infin"),
    ),
    AbbrsCol(
        ("subjunctive",),
        Abbr("sbjnct"),
    ),
    AbbrsCol(
        ("indicative",),
        Abbr("indic"),
    ),
    AbbrsCol(
        ("imperfect",),
        Abbr("impft"),
    ),
    
    # Animacy
    AbbrsCol(
        ("animate",),
        Abbr("ani"),
    ),
    AbbrsCol(
        ("inanimate",),
        Abbr("inani"),
    ),
    
    # Syntactic roles
    AbbrsCol(
        ("subject",),
        Abbr("sbj"),
    ),
    AbbrsCol(
        ("object",),
        Abbr("obj"),
    ),
    
    # Verb types
    AbbrsCol(
        ("reflexive",),
        Abbr("reflex"),
    ),
    AbbrsCol(
        ("transitive",),
        Abbr("trans"),
    )
]

def abbrev(inp: str, default: Union[str, None] = None, pref: str = "pref", chance: float = 1.0):
    """
    Find an abbreviation for the input string.
    
    Args:
        inp (str): The input string to abbreviate
        default (str, optional): Default value to return if no abbreviation is found.
                                If None, returns the original input string.
        pref (str): Determines which abbreviation to use:
                    - "pref": Use the preferred abbreviation (first one)
                    - "random": Randomly select from all abbreviations
                    - "least": Use the last abbreviation (least preferred)
        chance (float): Probability (0.0 to 1.0) of applying an abbreviation when found.
                       Default is 1.0 (always abbreviate).
    
    Returns:
        str: The selected abbreviation if found and chance roll succeeds,
             otherwise the default or input string
    
    Raises:
        ValueError: If pref is not one of the valid options ("pref", "random", "least")
    """
    # Validate pref parameter
    valid_prefs = ["pref", "random", "least"]
    if pref not in valid_prefs:
        raise ValueError(f"pref must be one of {valid_prefs}, got '{pref}'")
        
    # Set default to input if None
    if default is None:
        default = inp
        
    # Roll for chance to abbreviate if not 1.0
    if chance < 1.0 and random.random() > chance:
        return default
        
    # Convert input to lowercase for case-insensitive comparison
    inp_lower = inp.lower()
    
    # Check each AbbrsCol object
    for abbrev_col in abbrs:
        # Check if lowercase input matches any of the patterns (also lowercase)
        if any(pattern.lower() == inp_lower for pattern in abbrev_col.patterns):
            # Get the abbreviation based on pref parameter
            if pref == "pref":
                abbr_obj = abbrev_col.pref
            elif pref == "random":
                # Get all abbreviations and randomly select one
                abbr_obj = random.choice(abbrev_col.all())
            elif pref == "least":
                abbr_obj = abbrev_col.all()[-1]
                # print("least prefed", abbr_obj)
                
            
            # Get the text from the Abbr object
            result = abbr_obj.text
            
            # Apply capitalization rules
            if CapsRule.NoAll not in abbr_obj.caps_rules and inp.isupper():
                # If entire input is uppercase, make the abbreviation uppercase
                result = result.upper()
            elif CapsRule.NoFirst not in abbr_obj.caps_rules and len(inp) > 0 and inp[0].isupper():
                # If first character is uppercase, capitalize first character of abbreviation
                result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
            
            return result
    
    # If no match is found, return default
    return default

if __name__ == "__main__":
    # Test the abbrev function
    print(abbrev("javascript"))  # Should print "js"
    print(abbrev("Javascript", pref="least"))  # Should print "JS"
    print(abbrev("JAVASCRIPT"))  # Should print "JS"
    print(abbrev("python"))      # Should print "py"
    print(abbrev("unknown"))     # Should print "unknown"
    print(abbrev("unknown", default="not found"))  # Should print "not found"
    print(abbrev("something", pref="random"))  # Should print "s/t" (random pref but there is only one of them)