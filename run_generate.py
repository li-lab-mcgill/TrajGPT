"""Compatibility alias. The canonical entry point is now run_inference.py."""

from run_inference import (
    build_generation_input, generate_codes, main, parse_history, phenotype_names,
)


if __name__ == '__main__':
    main()
