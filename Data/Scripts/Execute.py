# Import required libraries
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

from plot_config import *
import subprocess
import datetime
import os

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

# Run 0_GetData.py
print(" Running 0_GetData.py...")
subprocess.run(["python", "./Scripts/0_GetData.py"])
print(f"0_GetData.py finished in {datetime.datetime.now() - start_time}.")
# Run 1_EDA.py
print(" Running 1_EDA.py...")
subprocess.run(["python", "./Scripts/1_EDA.py"])
print(f"1_EDA.py finished in {datetime.datetime.now() - start_time}.")
# Run 2_PreProcess.py
print(" Running 2_PreProcess.py...")
subprocess.run(["python", "./Scripts/2_PreProcess.py"])
print(f"2_PreProcess.py finished in {datetime.datetime.now() - start_time}.")
# Run 3_PLS.py
print(" Running 3_PLS.py...")
subprocess.run(["python", "./Scripts/3_PLS.py"])
print(f"3_PLS.py finished in {datetime.datetime.now() - start_time}.")

# Total runtime
print(f'Total runtime: {datetime.datetime.now() - start_time}.')

