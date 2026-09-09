"""
Create a non-overwriting Fig. 3c variant of the M3 technology-capacity matrix.

This wrapper reuses the current Nature-style plotting implementation and writes
to a separate fig3c_* filename so existing fig3b outputs remain unchanged.
"""

from m3_visualize_tech_capacity_supply import main


if __name__ == "__main__":
    main(
        output_stem="fig3c_tech_capacity_supply_matrix",
        source_prefix="fig3c",
    )
