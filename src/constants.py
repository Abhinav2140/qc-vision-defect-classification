"""Shared constants — kept dependency-free (no torch/cv2 imports) so that
lightweight tools like severity calibration, demo data generation, and the
dashboard export script can import them without pulling in the full ML stack."""

DEFECT_CLASSES = [
    "ok",
    "surface_scratch",
    "dent",
    "dimensional_error",
    "missing_component",
    "color_inconsistency",
    "contamination",
]
