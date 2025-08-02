# tabulapy/styling/styles.py
from abc import ABC, abstractmethod

# The data_formats module is needed for type checking and creating Value objects
from .. import data_formats

class StyleStrategy(ABC):
    """Abstract base class for all styling strategies."""
    @abstractmethod
    def apply(self, content, key_on: str = 'main') -> str:
        """
        Applies a LaTeX style to the given content.

        Args:
            content: The content to style (can be a Value object, str, etc.).
            key_on (str): The key that triggered the styling, for context.
        
        Returns:
            A LaTeX-formatted string.
        """
        pass

class BoldStyle(StyleStrategy):
    """
    Applies bold styling intelligently based on content type and styling key.
    """
    def apply(self, content, key_on: str = 'main') -> str:
        if isinstance(content, data_formats.Value):
            val = content
            main_formatted = f"{val.main:.{val.precision}f}"
            if val.pm is not None:
                pm_formatted = f"{val.pm:.{val.precision}f}"
                if key_on == 'sum':
                    return f"$\\boldsymbol{{{main_formatted} \\pm {pm_formatted}}}$"
                else: # 'main'
                    return f"$\\mathbf{{{main_formatted}}} \\pm {pm_formatted}$"
            else:
                return f"\\textbf{{{main_formatted}}}"
        return f"\\textbf{{{str(content)}}}"

class UnderlineStyle(StyleStrategy):
    """
    Applies underline styling intelligently based on content type and styling key.
    """
    def apply(self, content, key_on: str = 'main') -> str:
        # --- THIS IS THE NEW, INTELLIGENT LOGIC ---
        if isinstance(content, data_formats.Value):
            val = content
            main_formatted = f"{val.main:.{val.precision}f}"
            if val.pm is not None:
                pm_formatted = f"{val.pm:.{val.precision}f}"
                # If keyed on 'main', only underline the main part
                if key_on == 'main':
                    return f"$\\underline{{{main_formatted}}} \\pm {pm_formatted}$"
                # Otherwise (e.g., 'sum'), underline the whole expression for emphasis
                else:
                    return f"\\underline{{{str(val)}}}"
            else:
                # If no pm value, just underline the number
                return f"\\underline{{{main_formatted}}}"
        
        # Fallback for regular strings
        return f"\\underline{{{str(content)}}}"

class ItalicStyle(StyleStrategy):
    """Applies \\textit{...} styling. This style does not need context."""
    def apply(self, content, key_on: str = 'main') -> str:
        return f"\\textit{{{str(content)}}}"