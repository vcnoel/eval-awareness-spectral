Implemented the intervention package with NumPy-only synthetic tests.

- `patching.py`: validated layer/token/span/role targets, full-rank finite subspace bases, exact residual patching, rank-limited interchange, and four typed controls.
- `steering.py`: exact layer/token steering with explicit sufficiency and reverse-sign necessity directions.
- `mediation.py`: scientific endpoint interfaces, debug-only keyword rejection, behavioral sign-of-life gating, internal-first mediation assessment, and complete bidirectional/control validation.
- `__init__.py`: public exports for all intervention APIs.

Validation: **12 focused tests passed; 45 full-suite tests passed; Ruff passed.** Strict mypy passed under Python 3.12. The configured Python 3.11 mypy run was blocked by the installed NumPy stub using Python 3.12-only type syntax, not by project errors.