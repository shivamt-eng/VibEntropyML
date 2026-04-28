import pandas as pd
import numpy as np
import os
import phonopy
import gc
import time

# -----------------------------
# Setup paths (YOUR ORIGINAL LOGIC)
# -----------------------------
csv_filename = 'MPID_from_PhononDB_Database.csv'
base_db_path = r"/home/sudarshanhpc_see/phononDB/phonon_db"

# Constants
gas_constant_R = 8.314462618  # J/mol-K

start_time = time.time()

# Load MP IDs
df = pd.read_csv(csv_filename)
total_materials = len(df)

# Store results
results = []

for index, row in df.iterrows():
    material_id = row['material_id']

    # 🔹 Search inside database path (IMPORTANT FIX)
    matching_folders = [
        f for f in os.listdir(base_db_path)
        if material_id in f and os.path.isdir(os.path.join(base_db_path, f))
    ]

    if not matching_folders:
        print(f"[{index}] Skipped {material_id}: Folder not found in database.")
        continue

    # Take first match
    target_folder = os.path.join(base_db_path, matching_folders[0])

    # Full paths
    phonon_path = os.path.join(target_folder, "phonon.yaml")
    force_sets_path = os.path.join(target_folder, "FORCE_SETS")

    # 🔹 Critical file check (UNCHANGED LOGIC)
    if not os.path.exists(phonon_path) or not os.path.exists(force_sets_path):
        print(f"[{index}] Skipped {material_id}: Missing phonon.yaml or FORCE_SETS.")
        continue

    try:
        # Load phonopy
        ph = phonopy.load(
            phonon_path,
            force_sets_filename=force_sets_path,
            log_level=0
        )

        # Mesh
        ph.run_mesh([20, 20, 20])

        # 🔹 Temperature dependent calculation (MAIN CHANGE)
        ph.run_thermal_properties(t_min=100, t_max=500, t_step=50)

        thermal_dict = ph.get_thermal_properties_dict()

        temperatures = thermal_dict['temperatures']
        entropies = thermal_dict['entropy']

        num_atoms = len(ph.unitcell)

        # Store results
        for T, S in zip(temperatures, entropies):
            entropy_kB = (S / num_atoms) / gas_constant_R

            results.append({
                'mp-id': material_id,
                'temperature': T,
                'Svib_kB_atom': entropy_kB
            })

        # Cleanup
        del ph
        del thermal_dict

    except Exception as e:
        print(f"[{index}] Skipped {material_id} due to Phonopy error: {e}")
        continue

    gc.collect()

    # Save progress
    if (index + 1) % 100 == 0:
        pd.DataFrame(results).to_csv(
            "SVib_Temperature_Dependent.csv",
            index=False
        )
        print(f"Processed {index + 1}/{total_materials} materials...")

# Final save
pd.DataFrame(results).to_csv(
    "SVib_Temperature_Dependent.csv",
    index=False
)

end_time = time.time()
print(f"\nSUCCESS: Finished in {(end_time - start_time)/60:.2f} minutes")
