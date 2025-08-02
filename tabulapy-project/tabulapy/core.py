# tabulapy/core.py

import re
import copy
from .exceptions import *
from .styling import *
from .generator import LaTeXGenerator
from .data_formats import Value

class TabulaPy:
    _rule_map = {
        'highest': HighestValueRule,
        'second_highest': SecondHighestValueRule,
        'lowest': LowestValueRule,
    }
    _style_map = {
        'bold': BoldStyle,
        'underline': UnderlineStyle,
        'italic': ItalicStyle,
    }

    def __init__(self, headers, caption=None, label=None, alignments=None, adjustbox_width=None):
        if not headers:
            raise ConfigurationError("Headers list cannot be empty.")
        self.headers = headers
        self.caption = caption
        self.label = label
        self.adjustbox_width = adjustbox_width
        self.num_columns = len(headers)

        if alignments:
            if len(alignments) != self.num_columns:
                raise ConfigurationError("Length of alignments must match the number of headers.")
            self.alignments = alignments
        else:
            self.alignments = ['l'] + ['c'] * (self.num_columns - 1)

        self._data = []
        self._styling_jobs = []

    def add_row(self, row_data, row_color=None):
        if len(row_data) != self.num_columns:
            raise ValueError(f"Row data has {len(row_data)} items, but table has {self.num_columns} columns.")
        self._data.append({'data': row_data, 'color': row_color})

    def apply_column_style(self, column, rule: str, style: str, key_on: str = 'main'):
        col_idx = self._get_column_index(column)
        rule_class = self._rule_map.get(rule)
        if not rule_class: raise InvalidRuleException(f"Rule '{rule}' is not supported.")
        style_class = self._style_map.get(style)
        if not style_class: raise InvalidStyleException(f"Style '{style}' is not supported.")
        if key_on not in ['main', 'sum']: raise ValueError("key_on must be either 'main' or 'sum'.")

        self._styling_jobs.append({'col_idx': col_idx, 'rule': rule_class(), 'style': style_class(), 'key_on': key_on})

    def generate_tex(self, output_filename: str):
        processed_data = self._apply_styles()
        generator = LaTeXGenerator(
            processed_data=processed_data, headers=self.headers, alignments=self.alignments,
            caption=self.caption, label=self.label, adjustbox_width=self.adjustbox_width
        )
        final_tex = generator.generate()
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_tex)

    def _get_column_index(self, column):
        if isinstance(column, int):
            if 0 <= column < self.num_columns: return column
            raise ColumnNotFoundException(f"Column index {column} is out of range.")
        elif isinstance(column, str):
            try: return self.headers.index(column)
            except ValueError: raise ColumnNotFoundException(f"Column header '{column}' not found.")
        raise TypeError("Column must be an integer index or a string header.")

    @staticmethod
    def _parse_numeric(cell_value, key_on: str = 'main'):
        content = cell_value[0] if isinstance(cell_value, tuple) else cell_value
        if isinstance(content, Value):
            if key_on == 'sum' and content.pm is not None: return content.main + content.pm
            return content.main
        if isinstance(content, (int, float)): return float(content)
        if isinstance(content, str):
            match = re.search(r'[-+]?\d+(?:\.\d+)?', content)
            if match: return float(match.group(0))
        return None

    def _apply_styles(self):
        processed_data = copy.deepcopy(self._data)
        styles_to_apply = {}

        for job in self._styling_jobs:
            col_idx, rule_strategy, style_strategy, key_on = job.values()
            col_values = [v for row in self._data if (v := self._parse_numeric(row['data'][col_idx], key_on=key_on)) is not None]
            target_value = rule_strategy.find_target_value(col_values)
            if target_value is None: continue

            for row_idx, row in enumerate(self._data):
                if self._parse_numeric(row['data'][col_idx], key_on=key_on) == target_value:
                    if (row_idx, col_idx) not in styles_to_apply:
                        styles_to_apply[(row_idx, col_idx)] = []
                    # Store the strategy instance and the key for context
                    styles_to_apply[(row_idx, col_idx)].append((style_strategy, key_on))

        for (row_idx, col_idx), style_jobs in styles_to_apply.items():
            cell = processed_data[row_idx]['data'][col_idx]
            content, attrs = (cell, {}) if not isinstance(cell, tuple) else cell
            
            styled_content = content
            for style_strategy, key_on in style_jobs:
                styled_content = style_strategy.apply(styled_content, key_on=key_on)

            if isinstance(cell, tuple):
                processed_data[row_idx]['data'][col_idx] = (styled_content, attrs)
            else:
                processed_data[row_idx]['data'][col_idx] = styled_content

        return processed_data