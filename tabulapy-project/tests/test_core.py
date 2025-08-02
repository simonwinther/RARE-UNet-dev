# tests/test_core.py

import pytest
from tabulapy import TabulaPy, Value
from tabulapy.exceptions import ConfigurationError, ColumnNotFoundException

@pytest.fixture
def simple_table():
    headers = ["Metric", "Result"]
    table = TabulaPy(headers=headers, caption="Simple Table", label="tab:simple")
    table.add_row(["Accuracy", Value(0.98, precision=2)])
    table.add_row(["Dice Score", Value(0.85, pm=0.05)])
    table.apply_column_style("Result", "highest", "bold")
    return table

@pytest.fixture
def advanced_table():
    headers = ["Method", "DSC@res", "Avg. across scales"]
    alignments = ['L{2cm}', 'C{3cm}', 'c']
    table = TabulaPy(
        headers=headers, caption="Advanced Table", label="tab:advanced",
        alignments=alignments, adjustbox_width="\\textwidth"
    )
    table.add_row(
        ["RARE-UNet", Value(0.7147, pm=0.1462), Value(0.4411, pm=0.1725)],
        row_color="gray!15"
    )
    table.add_row(
        ["nnU-Net", Value(0.6050, pm=0.1725), Value(0.6050, precision=2, pm=0.1212)]
    )
    
    for column in ['DSC@res', 'Avg. across scales']:
        table.apply_column_style(column, 'highest', 'bold')
        table.apply_column_style(column, 'second_highest', 'underline')
    return table

def test_init_failure_mismatched_alignments():
    with pytest.raises(ConfigurationError):
        TabulaPy(headers=["A", "B"], alignments=['c', 'c', 'c'])

def test_add_row_failure_wrong_column_count():
    table = TabulaPy(headers=["A", "B"])
    with pytest.raises(ValueError):
        table.add_row(["cell1"])

def test_column_not_found():
    table = TabulaPy(headers=["A", "B"])
    with pytest.raises(ColumnNotFoundException):
        table.apply_column_style('C', 'highest', 'bold')

def test_simple_table_generation(simple_table, tmp_path):
    output_file = tmp_path / "simple.tex"
    simple_table.generate_tex(output_file)
    generated_content = output_file.read_text()
    expected_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage[table]{xcolor}
\usepackage{bm}

\begin{document}

\begin{table}[htbp]
\centering
\caption{Simple Table}
\label{tab:simple}
\begin{tabular}{lc}
\toprule
Metric & Result \\
\midrule
Accuracy & \textbf{0.98} \\
Dice Score & $0.8500 \pm 0.0500$ \\
\bottomrule
\end{tabular}
\end{table}

\end{document}
"""
    assert generated_content.strip() == expected_content.strip()

def test_advanced_table_generation(advanced_table, tmp_path):
    output_file = tmp_path / "advanced.tex"
    advanced_table.generate_tex(output_file)
    generated_content = output_file.read_text()
    expected_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage[table]{xcolor}
\usepackage{bm}
\usepackage{adjustbox}
\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

\begin{document}

\begin{table}[htbp]
\centering
\caption{Advanced Table}
\label{tab:advanced}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{L{2cm}C{3cm}c}
\toprule
Method & DSC@res & Avg. across scales \\
\midrule
\rowcolor{gray!15} RARE-UNet & $\mathbf{0.7147} \pm 0.1462$ & $\underline{0.4411} \pm 0.1725$ \\
nnU-Net & $\underline{0.6050} \pm 0.1725$ & $\mathbf{0.61} \pm 0.12$ \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\end{document}
"""
    assert generated_content.strip() == expected_content.strip()

def test_styling_granularity(tmp_path):
    table = TabulaPy(headers=["Model", "Score (Main)", "Score (Sum)"])
    table.add_row(["A", Value(0.9, pm=0.01), Value(0.9, pm=0.01)])
    table.add_row(["B", Value(0.8, pm=0.20), Value(0.8, pm=0.20)])
    
    table.apply_column_style("Score (Main)", "highest", "bold", key_on='main')
    table.apply_column_style("Score (Sum)", "highest", "bold", key_on='sum')
    
    output_file = tmp_path / "granularity.tex"
    table.generate_tex(output_file)
    generated = output_file.read_text()
    
    # Check for \mathbf on Model A in the "Main" column
    assert r"A & $\mathbf{0.9000} \pm 0.0100$ &" in generated
    # Check for \boldsymbol on Model B in the "Sum" column
    assert r"& $\boldsymbol{0.8000 \pm 0.2000}$" in generated