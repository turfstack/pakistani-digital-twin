# Import the "sys" module, which lets us read information about Python itself
import sys

# Import five libraries we installed; if any is missing, Python stops here with an error that tells us which one
import numpy, pandas, scipy, sklearn, matplotlib

# Print the Python version; sys.version is a long string, so we split it at spaces and keep only the first piece
print("Python:", sys.version.split()[0])

# Loop over each library one at a time, calling the current one "m"
for m in (numpy, pandas, scipy, sklearn, matplotlib):
    # Print the library's name and its version number
    print(m.__name__, m.__version__)

# "try" means: attempt the next lines, and if they fail, jump to "except" instead of crashing
try:
    # Import torch (PyTorch), the deep learning library we will use for the AI twin
    import torch
    # If the import worked, print its version
    print("torch", torch.__version__)
# This runs only if the import above failed because torch isn't installed
except ImportError:
    # Tell us torch is missing so we know to fix it
    print("torch NOT installed yet")