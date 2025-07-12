import sys, os

def add_project_root(levels_up=5):
    current_file = os.path.abspath(__file__)
    root = current_file
    for _ in range(levels_up):
        root = os.path.dirname(root)
    if root not in sys.path:
        sys.path.insert(0, root)
