import logging
import matplotlib.font_manager as fm
import matplotlib as mpl
from typing import Dict, Optional
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def setup_font(font_path: Optional[str], font_size: Optional[float] = None):
    if font_path and os.path.exists(font_path):
        try:
            logger.info(f"Setting global font to: {font_path}")
            font_prop = fm.FontProperties(fname=font_path)
            fm.fontManager.addfont(font_path)
            mpl.rcParams['font.family'] = font_prop.get_name()
            mpl.rcParams['pdf.fonttype'] = 42
            mpl.rcParams['ps.fonttype'] = 42
            if font_size:
                mpl.rcParams['font.size'] = font_size
                logger.info(f"Set global font size to: {font_size}")
        except Exception as e:
            logger.warning(f"Could not set custom font: {e}. Using Matplotlib's default font.")
    elif font_path:
        logger.warning(f"Font file not found at '{font_path}'. Using Matplotlib's default font.")