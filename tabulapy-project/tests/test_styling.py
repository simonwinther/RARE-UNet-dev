# tests/test_styling.py

import pytest
from tabulapy.styling.rules import HighestValueRule, SecondHighestValueRule, LowestValueRule
from tabulapy.styling.styles import BoldStyle, UnderlineStyle
from tabulapy.core import TabulaPy  # For accessing the private _parse_numeric

# --- Test Rule Strategies ---

@pytest.mark.parametrize("rule_class, numbers, expected", [
    (HighestValueRule, [1, 5, 3], 5),
    (HighestValueRule, [10.5, 10.1, 10.9], 10.9),
    (HighestValueRule, [5, 5, 5], 5),
    (HighestValueRule, [], None),
    (SecondHighestValueRule, [1, 5, 3], 3),
    (SecondHighestValueRule, [10, 20, 30, 40], 30),
    (SecondHighestValueRule, [10, 20, 20, 10], 10),
    (SecondHighestValueRule, [5], None),
    (LowestValueRule, [1, 5, 3], 1),
    (LowestValueRule, [-10, 0, 10], -10),
])
def test_rule_strategies(rule_class, numbers, expected):
    """Tests that all rule strategies find the correct target value."""
    rule = rule_class()
    assert rule.find_target_value(numbers) == expected

# --- Test Style Strategies ---

def test_bold_style():
    """Tests the bold style strategy."""
    style = BoldStyle()
    assert style.apply("hello") == "\\textbf{hello}"

def test_underline_style():
    """Tests the underline style strategy."""
    style = UnderlineStyle()
    assert style.apply("world") == "\\underline{world}"

# --- Test Utility Functions (like _parse_numeric) ---

@pytest.mark.parametrize("input_str, expected_float", [
    ("0.7165", 0.7165),
    ("$0.7165 \\pm$ 0.1489", 0.7165),
    ("-5.2 apples", -5.2),
    ("No number here", None),
    (123, 123.0),
    (("0.5", {}), 0.5), # Test with a tuple
    (None, None)
])
def test_parse_numeric(input_str, expected_float):
    """Tests the internal numeric parser with various string formats."""
    assert TabulaPy._parse_numeric(input_str) == expected_float