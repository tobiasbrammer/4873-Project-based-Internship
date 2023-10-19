# Import required libraries
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

from plot_config import *
import subprocess
import datetime
import os
import winsound

# Add plot_config.py to env variables
os.environ['PYTHONPATH'] = os.getcwd()

# Start timing
start_time = datetime.datetime.now()

# Set the directory
directory = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
os.chdir(directory)

# Clear console
os.system('cls')

#### INPUT DEPENDENT VARIABLE ####
sDepVar = 'contribution'

# Save depvar to ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'w') as f:
    f.write(sDepVar)

print(f"Dependent variable: {sDepVar}")
print('Preparing to run sequence of scripts...')
# Run 0_GetData.py
print("Running 0_GetData.py...")
subprocess.run(["python", "./Scripts/0_GetData.py"])

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


# Total runtime
print(f'Execution finished in {datetime.datetime.now() - start_time}.')

# Play sound when finished
winsound.MessageBeep()
