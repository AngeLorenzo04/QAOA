import sys
import os

# Aggiunge la directory src/ al path di Python
# in modo che pytest possa trovare i pacchetti max_cut, max_k_cut, ecc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
