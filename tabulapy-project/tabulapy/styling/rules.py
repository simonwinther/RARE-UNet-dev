# tabulapy/styling/rules.py
from abc import ABC, abstractmethod
from typing import List, Optional, Union

class RuleStrategy(ABC):
    """Abstract base class for all rule-finding strategies."""
    @abstractmethod
    def find_target_value(self, values: List[Union[int, float]]) -> Optional[Union[int, float]]:
        """
        Finds the target value from a list of numbers based on the rule.
        Returns None if a target cannot be found.
        """
        pass

class HighestValueRule(RuleStrategy):
    """Finds the maximum value."""
    def find_target_value(self, values: List[Union[int, float]]) -> Optional[Union[int, float]]:
        unique_sorted = sorted(list(set(values)))
        return unique_sorted[-1] if unique_sorted else None

class SecondHighestValueRule(RuleStrategy):
    """Finds the second-highest unique value."""
    def find_target_value(self, values: List[Union[int, float]]) -> Optional[Union[int, float]]:
        unique_sorted = sorted(list(set(values)))
        return unique_sorted[-2] if len(unique_sorted) > 1 else None

class LowestValueRule(RuleStrategy):
    """Finds the minimum value."""
    def find_target_value(self, values: List[Union[int, float]]) -> Optional[Union[int, float]]:
        unique_sorted = sorted(list(set(values)))
        return unique_sorted[0] if unique_sorted else None