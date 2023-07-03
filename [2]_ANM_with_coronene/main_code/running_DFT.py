import sys
sys.path.append('../..')
sys.path.append('../../APDFT')
sys.path.append('../../helper_code')
sys.path.append('../Data')

import pickle
from pyscf import gto, scf, dft, cc
import numpy as np
import pandas as pd
import pyscf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import basis_set_exchange as bse
from APDFT.FcMole import *
import os
import ast
from copy import deepcopy
from IPython.display import display
from helper_code.data_processing import *
from APDFT.AP_class import APDFT_perturbator as AP

coronene_atom = """
C        1.02559207      -3.63123345      -0.00000000
C        2.25864795      -3.02257024      -0.00000004
C        2.38243776      -1.59070223       0.00000001
C        1.19669491      -0.79904388       0.00000001
C       -0.09357741      -1.43595331      -0.00000003
C       -0.18628682      -2.85866989      -0.00000002
C       -1.48823293      -3.46741915       0.00000006
C       -2.63182057      -2.70389221      -0.00000001
C       -2.56872591      -1.26805944      -0.00000005
C       -1.29025590      -0.63698042      -0.00000004
C       -1.19669715       0.79886895      -0.00000003
C       -2.38243735       1.59053206      -0.00000010
C       -3.65743830       0.92722391       0.00000002
C       -3.74684719      -0.44495256      -0.00000001
C       -2.25864027       3.02239662      -0.00000008
C       -1.02558026       3.63105875      -0.00000009
C        0.18629225       2.85849513      -0.00000001
C        0.09357535       1.43577833       0.00000001
C        1.29025258       0.63680458       0.00000002
C        2.56871742       1.26789623      -0.00000004
C        3.74683463       0.44479444      -0.00000003
C        3.65742920      -0.92738538      -0.00000004
C        2.63181870       2.70373227       0.00000003
C        1.48823680       3.46725735       0.00000009
H        0.94770675      -4.72429028       0.00000003
H        3.17370895      -3.62548407       0.00000000
H       -1.55279282      -4.56134681       0.00000003
H       -3.61740441      -3.18291867       0.00000006
H       -4.56511975       1.54119766       0.00000009
H       -4.72651664      -0.93596671       0.00000001
H       -3.17369776       3.62531592       0.00000001
H       -0.94769291       4.72411558       0.00000002
H        4.72650482       0.93580817       0.00000006
H        4.56511650      -1.54134951      -0.00000004
H        3.61740413       3.18275567       0.00000004
H        1.55279359       4.56118658       0.00000004
"""

if __name__ == "__main__":
    basis_pcx2={"H": "pc-2", 'C': bse.get_basis("pcX-2", fmt="nwchem", elements=[6])}
    mol = gto.M(atom=coronene_atom, basis=basis_pcx2, unit='Angstrom')
    mol_DFT = scf.RKS(mol)
    mol_DFT.xc = "PBE0" # specify the exchange-correlation functional used for DFT
    mol_DFT.kernel() # run self-consistent field calculation
    mol_total_energy = mol_DFT.energy_tot()
    mol_electronic_energy = mol_DFT.energy_elec()
    print(mol_total_energy)
    print(mol_electronic_energy)
