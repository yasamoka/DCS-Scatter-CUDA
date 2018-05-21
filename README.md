CUDA Implementation of the Direct Coulomb Summation (DCS) scatter & gather algorithms.

Tested using Visual Studio 2017 (v141) and CUDA v9.1.

Example run:

Parameters: width = 50, height = 50, depth = 50, grid spacing = 0.1, number of atoms = 5000, maximum charge = 10, number of streams = 1

Commandline:
```
DCS-Scatter-CUDA -x 50 -y 50 -z 50 -g 0.1 -a 5000 -c 10 -n 1
DCS-Gather-CUDA -x 50 -y 50 -z 50 -g 0.1 -a 5000 -c 10 -n 1
```