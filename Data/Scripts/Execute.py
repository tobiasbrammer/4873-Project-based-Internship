# Import required libraries
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Set the directory
import os
if os.name == 'posix':
    sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
# If operating system is Windows then
elif os.name == 'nt':
    sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

os.chdir(sDir)

import subprocess
import datetime
from plot_config import *

# Add plot_config.py to env variables
os.environ['PYTHONPATH'] = os.getcwd()

# Start timing
start_time = datetime.datetime.now()

#### INPUT DEPENDENT VARIABLE ####
sDepVar = 'total_contribution'
trainMethod = 'train'

# Save depvar to ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'w') as f:
    f.write(sDepVar)

with open('./.AUX/trainMethod.txt', 'w') as f:
    f.write(trainMethod)

print(f"Dependent variable: {sDepVar}")
print('Preparing to run sequence of scripts...')
# Run 0_GetData.py
print("Running 0_GetData.py...")
#subprocess.run(["python", "./Scripts/0_GetData.py"])

# Run 1_EDA.py
print("Running 1_EDA.py...")
subprocess.run(["python", "./Scripts/1_EDA.py"])

# Run 2_PreProcess.py
print("Running 2_PreProcess.py...")
subprocess.run(["python", "./Scripts/2_PreProcess.py"])

# Run 3_PLS.py
print("Running 3_PLS.py...")
subprocess.run(["python", "./Scripts/3_PLS.py"])

# Run 4_ML.py
print("Running 4_ML.py...")
subprocess.run(["python", "./Scripts/4_ML.py"])

# Run 5_DL.py
print("Running 5_DL.py...")
subprocess.run(["python", "./Scripts/5_DL.py"])

# Total runtime
print(f'Execution finished in {datetime.datetime.now() - start_time}.')

# Play sound when finished
if os.name == 'posix':
    os.system('say "Finished.')
# If operating system is Windows then
elif os.name == 'nt':
    import winsound
    winsound.Beep(frequency=600, duration=800)

