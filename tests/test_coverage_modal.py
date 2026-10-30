"""C0: modal_jobs/coverage.py must build on the exact image chain of reconstruct.py, so
Modal reuses the cached gsplat layer instead of compiling gsplat again (41 min)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _chain(text: str) -> str:
    start = text.index("# Only the GPU the job runs on")
    end = text.index("    # ---- END copy") if "# ---- END copy" in text else None
    if end is None:
        end = text.index('    .add_local_python_source("movable_reality"')
    return text[start:end]


def test_coverage_image_is_a_byte_copy_of_the_reconstruct_image():
    rec = (ROOT / "modal_jobs" / "reconstruct.py").read_text()
    cov = (ROOT / "modal_jobs" / "coverage.py").read_text()
    assert _chain(rec) == _chain(cov)
    # Coverage-only layers come after the copy.
    assert cov.index("# ---- END copy") < cov.index('.pip_install("plyfile')
