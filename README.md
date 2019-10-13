# Spherical-Principal-Curve
Reproducible Code for Spherical Principal Curve

# Requirements
```
Python 3.*
numpy 1.13.3 +
torch 1.1.0 +
pandas
openpyxl
```

# Visualization
Check Final Code.IPYNB (install jupyter notebook and execute 'jupyter notebook' in CMD)

# Python Codes
- utils.py: function modules
- principal_curves.py: codes for principal curve fitting
- main.py: For Earthquake data (fitted curve is saved)
```
python main.py -q 0.05 -e True -i False 
Note, q : neighborhood ratio, e : ours (True) or Hauberg's(False), i : intrinsic (True) or extrinsic (False)
```
- silmulation.py: simulation data experiment (excel files are saved):
```
python simulation.py -q 0.05 -n 100
```
