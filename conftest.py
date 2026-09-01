"""
Muss vor dem ersten numpy-Import in irgendeinem Testmodul laufen (pytest
sammelt conftest.py vor den eigentlichen Testdateien ein) - siehe Kommentar
in app.py: OpenBLAS liest die Thread-Anzahl beim Laden der Bibliothek, eine
spaeter gesetzte Umgebungsvariable wirkt nicht mehr.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
