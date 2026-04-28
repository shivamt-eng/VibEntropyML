import os
import glob
import pandas as pd
import phonopy
import numpy as np
import gc
import time

# 1. Setup your paths based on your screenshots
csv_filename = 'MPID_from_PhononDB_Database.csv'
# Replace this with the exact path if your jupyter notebook is not in the 'database2' folder
base_db_path = r"/home/sudarshanhpc_see/phononDB/phonon_db" 

# Load the dataset
df = pd.read_csv(csv_filename)
df['Svib_300K_kB'] = np.nan

gas_constant_R =  8.31446261815324
total_materials = len(df)

print(f"Starting calculation for {total_materials} materials...")
start_time = time.time()

for index, row in df.iterrows():
    # Grab the ID, e.g., "mp-1000"
    material_id = str(row['material_id']).strip()
    
    # Search for the folder that starts with this material_id 
    # The '*' acts as a wildcard for the date suffix
    search_pattern = os.path.join(base_db_path, f"{material_id}-*")
    matching_folders = glob.glob(search_pattern)
    
    # Check if the folder actually exists
    if not matching_folders:
        print(f"[{index}] Skipped {material_id}: Folder not found.")
        continue
        
    # Take the first matched folder
    target_folder = matching_folders[0]
    
    # Construct the exact file paths
    phonon_path = os.path.join(target_folder, "phonon.yaml")
    force_sets_path = os.path.join(target_folder, "FORCE_SETS")
    
    # Double check that the necessary files are inside the folder
    if not os.path.exists(phonon_path) or not os.path.exists(force_sets_path):
        print(f"[{index}] Skipped {material_id}: Missing phonon.yaml or FORCE_SETS in folder.")
        continue

    try:
        # Load silently
        ph = phonopy.load(phonon_path, force_sets_filename=force_sets_path, log_level=0)
        
        # Calculate at 300K using the faster 20x20x20 mesh
        ph.run_mesh([20, 20, 20]) 
        ph.run_thermal_properties(t_min=300, t_max=300, t_step=1)
        
        # Extract and convert
        thermal_dict = ph.get_thermal_properties_dict()
        raw_entropy = thermal_dict['entropy'][0]
        
        num_atoms = len(ph.unitcell)
        entropy_kB = (raw_entropy / num_atoms) / gas_constant_R
        
        # Store in dataframe
        df.at[index, 'Svib_300K_kB'] = entropy_kB
        
        # Free up memory
        del ph
        del thermal_dict
        
    except Exception as e:
        print(f"[{index}] Skipped {material_id} due to Phonopy calculation error: {e}")
        pass 
    
    # Force memory cleanup
    gc.collect()
    
    # Save a backup every 100 items
    if (index + 1) % 100 == 0:
        df.to_csv('SVib_kB_PhononDB.csv', index=False)
        print(f"Processed {index + 1}/{total_materials}...")

# Final Save
df.to_csv('SVib_kB_PhononDB.csv', index=False)

end_time = time.time()
print(f"\nSUCCESS: Finished processing all materials in {(end_time - start_time)/60:.2f} minutes.")
