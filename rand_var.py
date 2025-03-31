import random

class RandVar:
    def __init__(self, min_val, max_val, val=None):
        """
        Initialize a random variable with a minimum and maximum value.
        
        Args:
            min_val: The minimum possible value (int or float)
            max_val: The maximum possible value (int or float)
            val: Initial value (optional). If not provided, a random value is generated.
        """
        self.min = min_val
        self.max = max_val
        
        if self.min > self.max:
            raise ValueError("Minimum value cannot be greater than maximum value")
    
        # Set initial value
        if val is not None:
            if not (isinstance(val, (int, float))):
                raise TypeError("Value must be an integer or float")
            if val < self.min or val > self.max:
                raise ValueError(f"Value must be between {self.min} and {self.max}")
            self.val = val
        else:  # no val passed
            if isinstance(self.min, int) and isinstance(self.max, int):
                self.val = 0
            else:
                self.val = 0.0
            # self.roll()
    
    def roll(self):
        """
        Reroll the value according to min and max.
        For integers, use randint. For floats, use uniform.
        
        Returns:
            The new value after rerolling
        """
        # Check if both min and max are integers
        if isinstance(self.min, int) and isinstance(self.max, int):
            self.val = random.randint(self.min, self.max)
        else:
            self.val = random.uniform(self.min, self.max)
        
        return self.val
    
    def __repr__(self):
        """String representation of the RandVar"""
        return f"RandVar(min={self.min}, max={self.max}, val={self.val})"