CUDA Implementation of the Direct Coulomb Summation (DCS) Scatter algorithm.

Tested using Visual Studio 2017 (v141).

Example run:

Parameters: width = 5, height = 5, depth = 5, grid spacing = 0.1, number of atoms = 5000, maximum charge = 10, number of streams = 8

Commandline:
```
DCS-Scatter-CUDA -x 5 -y 5 -z 5 -g 0.1 -a 5000 -c 10 -n 8
```