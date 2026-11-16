"""Run the scored CLI workflow with a scripted local provider.

PYTHONPATH=src python3 references/examples/end_to_end.py --mode single
PYTHONPATH=src python3 references/examples/end_to_end.py --mode multi
"""
from unified_solving import main

if __name__ == "__main__":
    main()
