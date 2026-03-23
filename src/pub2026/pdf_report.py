"""PDF report generator with annotated pages including timestamps and source paths."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src.visualise.settings import apply_mpl_settings


class PDFReport:
    """Multi-page PDF report with automatic annotations on each page.

    Every page includes a footer with:
    - Generation timestamp
    - Source data file path(s)
    - Config YAML file path (if provided)
    """

    def __init__(self,
                 output_path: str,
                 title: str = "Analysis Report",
                 config_path: Optional[str] = None):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.config_path = config_path
        self.generation_time = datetime.now()
        self._pdf = PdfPages(str(self.output_path))
        self._page_count = 0
        apply_mpl_settings()
        self._add_title_page()

    def _format_timestamp(self) -> str:
        return self.generation_time.strftime("%Y-%m-%d %H:%M:%S")

    def _add_footer(self,
                    fig: Figure,
                    source_paths: Optional[List[str]] = None):
        """Add annotation footer to a figure."""
        footer_parts = [f"Generated: {self._format_timestamp()}"]
        if self.config_path:
            footer_parts.append(f"Config: {self.config_path}")
        if source_paths:
            paths_str = ", ".join(str(p) for p in source_paths)
            footer_parts.append(f"Source data: {paths_str}")
        footer_parts.append(f"Page {self._page_count}")

        footer_text = "  |  ".join(footer_parts)
        fig.text(0.5,
                 0.01,
                 footer_text,
                 ha='center',
                 va='bottom',
                 fontsize=6,
                 color='gray',
                 style='italic',
                 transform=fig.transFigure)

    def _add_title_page(self):
        """Add a title page with report metadata."""
        fig = plt.figure(figsize=(10, 7))
        self._page_count += 1

        fig.text(0.5,
                 0.65,
                 self.title,
                 ha='center',
                 va='center',
                 fontsize=20,
                 fontweight='bold')
        fig.text(0.5,
                 0.55,
                 f"Generated: {self._format_timestamp()}",
                 ha='center',
                 va='center',
                 fontsize=12,
                 color='gray')

        info_lines = []
        if self.config_path:
            info_lines.append(f"Configuration: {self.config_path}")
        info_lines.append(f"Output: {self.output_path}")

        info_text = "\n".join(info_lines)
        fig.text(0.5,
                 0.40,
                 info_text,
                 ha='center',
                 va='center',
                 fontsize=10,
                 family='monospace')

        self._add_footer(fig)
        self._pdf.savefig(fig)
        plt.close(fig)

    def add_figure(self,
                   fig: Figure,
                   caption: str = "",
                   source_paths: Optional[List[str]] = None):
        """Add a matplotlib figure as a new page.

        Parameters
        ----------
        fig : Figure
            The matplotlib figure to add.
        caption : str
            Caption text shown below the figure.
        source_paths : list of str, optional
            Data file paths used to create this figure.
        """
        self._page_count += 1

        # Add major grid to all axes
        for ax in fig.get_axes():
            ax.grid(True, which='major', alpha=0.3)

        if caption:
            fig.text(0.5,
                     0.02,
                     caption,
                     ha='center',
                     va='bottom',
                     fontsize=8,
                     style='italic',
                     wrap=True)

        self._add_footer(fig, source_paths)
        self._pdf.savefig(fig)
        plt.close(fig)

    def add_table(self,
                  df: pd.DataFrame,
                  title: str = "",
                  source_paths: Optional[List[str]] = None):
        """Add a DataFrame as a table on a new page.

        Parameters
        ----------
        df : DataFrame
            Data to render as a table.
        title : str
            Title above the table.
        source_paths : list of str, optional
            Data file paths used.
        """
        self._page_count += 1
        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(df) + 2)))
        ax.axis('off')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        cell_text = df.values.tolist()
        col_labels = list(df.columns)

        table = ax.table(cellText=cell_text,
                         colLabels=col_labels,
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Style header row
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')

        self._add_footer(fig, source_paths)
        self._pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def add_text(self,
                 text: str,
                 title: str = "",
                 source_paths: Optional[List[str]] = None):
        """Add a text page.

        Parameters
        ----------
        text : str
            Body text content.
        title : str
            Title at top of page.
        source_paths : list of str, optional
            Data file paths referenced.
        """
        self._page_count += 1
        fig = plt.figure(figsize=(10, 7))

        if title:
            fig.text(0.5,
                     0.92,
                     title,
                     ha='center',
                     va='top',
                     fontsize=14,
                     fontweight='bold')

        fig.text(0.1,
                 0.85,
                 text,
                 ha='left',
                 va='top',
                 fontsize=10,
                 family='monospace',
                 wrap=True,
                 transform=fig.transFigure)

        self._add_footer(fig, source_paths)
        self._pdf.savefig(fig)
        plt.close(fig)

    def add_metrics(self,
                    metrics: Dict[str, Dict[str, Union[float, str]]],
                    title: str = "Metrics Summary",
                    source_paths: Optional[List[str]] = None):
        """Add a metrics comparison table.

        Parameters
        ----------
        metrics : dict
            {dataset_name: {metric_name: value}}
        title : str
            Table title.
        source_paths : list of str, optional
            Data file paths used.
        """
        df = pd.DataFrame(metrics).T
        df.index.name = "Dataset"
        df = df.reset_index()
        self.add_table(df, title=title, source_paths=source_paths)

    def save(self):
        """Close and save the PDF."""
        self._pdf.close()
        return str(self.output_path)
