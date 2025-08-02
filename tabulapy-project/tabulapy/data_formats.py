# tabulapy/data_formats.py

from typing import Optional

class Value:
    """
    Represents a numerical value in a table cell, handling formatting.

    This class provides a data-centric way to define cell content, separating
    the raw numbers from their LaTeX representation.
    """
    def __init__(self, main: float, pm: Optional[float] = None, precision: int = 4):
        """
        Args:
            main (float): The primary numerical value (e.g., mean). This is used
                for sorting and conditional styling rules.
            pm (float, optional): The plus-minus value (e.g., standard deviation).
                If provided, formats as 'main ± pm'. Defaults to None.
            precision (int, optional): The number of decimal places for formatting.
                Defaults to 4.
        """
        if not isinstance(main, (int, float)):
            raise TypeError("main value must be a number.")
        if pm is not None and not isinstance(pm, (int, float)):
            raise TypeError("pm value must be a number.")
            
        self.main = main
        self.pm = pm
        self.precision = precision

    def __str__(self) -> str:
        """Generates the LaTeX string representation of the value."""
        main_formatted = f"{self.main:.{self.precision}f}"
        
        if self.pm is not None:
            pm_formatted = f"{self.pm:.{self.precision}f}"
            return f"${main_formatted} \\pm {pm_formatted}$"
        else:
            return main_formatted

    def __repr__(self) -> str:
        """Provides an unambiguous representation for debugging."""
        if self.pm is not None:
            return f"Value({self.main}, pm={self.pm}, precision={self.precision})"
        return f"Value({self.main}, precision={self.precision})"