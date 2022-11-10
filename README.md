**Required Python version:**
Python 3 

**Code Structure:**

1) Importation of the required Python libraries

(numpy,
matplotlib.pyplot,
math,
random,
sys,
pandas,
torch and
igraph)

2) Model parameters

Here, the user can set particles parameters, the composition of the binary mix, the simulation time,parameters describing the simulation space, as well as  the number of timesteps at which snapshot of the systems are saved and order parameters are measured during the simulation.

_NB:_ Changes in particles packing fraction induce changes in the size of the simulation space. Such changes are constrained by the number of boxes used in the simulation. The actual packing fraction, that might differs from the desired value set by the user, is displayed.

3) Definition of Torch molecular simulation functions

4) Initialization of the system according to the parameters set in 1)

5) System evolution

6) This part of the code is dedicated to the calculation of a set of order parameters describing the final state of the system.





