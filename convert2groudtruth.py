"""
Goal:
    using DSPY to convert backend data to frontend. 
    That is, to convert input columns and rows to their ground truth counterparts.
"""
from typing import Dict

def generate_column_mapping(json_path: str) -> Dict[int, int]:
    """
    Generate a mapping that map the backend columns
    to the frontend columns
    """