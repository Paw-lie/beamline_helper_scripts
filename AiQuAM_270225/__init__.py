"""
AiQuAM - AI Quality Assessment Module
"""

# Make Q3 modules easily importable
from . import Q3_open_source_image_processing_functions
from . import Q3_open_source_fit_two_half_gaussians

# Import Q3_dev_workspace
try:
    from . import Q3_dev_workspace
except (ImportError, SyntaxError) as e:
    import warnings
    warnings.warn(f"Could not import Q3_dev_workspace: {e}")
    Q3_dev_workspace = None

__all__ = [
    'Q3_open_source_image_processing_functions',
    'Q3_open_source_fit_two_half_gaussians',
    'Q3_dev_workspace'
]
