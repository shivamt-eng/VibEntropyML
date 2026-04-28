SVib_kB_PhononDB.py:
Computes vibrational entropy at 300 K using Phonopy from PhononDB inputs (FORCE_SETS and phonopy.yaml). The calculation uses a 20x20x20 q-point mesh and reports entropy in units of kB/atom within the harmonic approximation.

SVib_kB_PhononDB.csv:
Dataset of vibrational entropy values (kB/atom) at 300 K derived from PhononDB and mapped to Materials Project IDs.

SVib_kB_PhononDB_T.py:
Computes vibrational entropy over the range 100–500 K in steps of 50 K using Phonopy from PhononDB inputs (FORCE_SETS and phonopy.yaml). The calculation uses a 20x20x20 q-point mesh and reports entropy in units of kB/atom within the harmonic approximation.

SVib_Temperature_Dependent.csv:
Temperature-dependent vibrational entropy dataset (kB/atom) spanning 100–500 K (50 K intervals), mapped to Materials Project IDs.

ML_predicted_Svib.ipynb:
Notebook implementing the machine learning pipeline for vibrational entropy prediction, including feature construction, model training, and evaluation.

