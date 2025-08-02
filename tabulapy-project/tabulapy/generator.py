# tabulapy/generator.py

class LaTeXGenerator:
    """Generates a complete LaTeX document from processed table data."""

    def __init__(self, processed_data, headers, alignments, caption, label, adjustbox_width):
        self.data = processed_data
        self.headers = headers
        self.alignments = alignments
        self.caption = caption
        self.label = label
        self.adjustbox_width = adjustbox_width

    def generate(self) -> str:
        """Builds the final LaTeX string."""
        tex_parts = []
        self._add_preamble(tex_parts)
        
        tex_parts.append("")
        tex_parts.append("\\begin{document}")
        tex_parts.append("")

        self._add_table_environment(tex_parts)
        
        tex_parts.append("")
        tex_parts.append("\\end{document}")
        tex_parts.append("")

        return "\n".join(tex_parts)

    def _add_preamble(self, tex_parts):
        tex_parts.append("\\documentclass{article}")
        tex_parts.append("\\usepackage[utf8]{inputenc}")
        tex_parts.append("\\usepackage{booktabs}")
        tex_parts.append("\\usepackage{array}")
        tex_parts.append("\\usepackage{multirow}")
        tex_parts.append("\\usepackage[table]{xcolor}")
        
        if self.adjustbox_width:
            tex_parts.append("\\usepackage{adjustbox}")
        
        alignment_str = "".join(self.alignments)
        if 'L{' in alignment_str:
            tex_parts.append("\\newcolumntype{L}[1]{>{\\raggedright\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}")
        if 'C{' in alignment_str:
            tex_parts.append("\\newcolumntype{C}[1]{>{\\centering\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}")
        if 'R{' in alignment_str:
            tex_parts.append("\\newcolumntype{R}[1]{>{\\raggedleft\\let\\newline\\\\\\arraybackslash\\hspace{0pt}}m{#1}}")

    def _add_table_environment(self, tex_parts):
        tex_parts.append("\\begin{table}[htbp]")
        tex_parts.append("\\centering")
        
        if self.caption:
            tex_parts.append(f"\\caption{{{self.caption}}}")
        if self.label:
            tex_parts.append(f"\\label{{{self.label}}}")
            
        if self.adjustbox_width:
            tex_parts.append(f"\\begin{{adjustbox}}{{max width={self.adjustbox_width}}}")

        tex_parts.append(f"\\begin{{tabular}}{{{''.join(self.alignments)}}}")
        tex_parts.append("\\toprule")
        tex_parts.append(" & ".join(self.headers) + " \\\\")
        tex_parts.append("\\midrule")
        
        for row_dict in self.data:
            row_cells = []
            for cell in row_dict['data']:
                if cell is None:
                    row_cells.append("")
                    continue

                content, attrs = (cell, {}) if not isinstance(cell, tuple) else cell
                tex_cell = str(content)
                
                # --- THIS IS THE FIX ---
                # Only generate a \multirow command if the span is actually greater than 1.
                rowspan_val = attrs.get("rowspan")
                if rowspan_val and rowspan_val > 1:
                    tex_cell = f"\\multirow{{{rowspan_val}}}{{*}}{{{tex_cell}}}"
                # --- END OF FIX ---
                    
                if attrs.get("colspan"):
                    tex_cell = f"\\multicolumn{{{attrs['colspan']}}}{{{attrs.get('align', 'c')}}}{{{tex_cell}}}"
                row_cells.append(tex_cell)
            
            row_str = " & ".join(row_cells) + " \\\\"
            if row_dict['color']:
                row_str = f"\\rowcolor{{{row_dict['color']}}} " + row_str
            tex_parts.append(row_str)
            
        tex_parts.append("\\bottomrule")
        tex_parts.append("\\end{tabular}")
        
        if self.adjustbox_width:
            tex_parts.append("\\end{adjustbox}")
            
        tex_parts.append("\\end{table}")