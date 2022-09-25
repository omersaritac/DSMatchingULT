## Matching Policies
The code includes Python code for the OR paper "Dynamic Stochastic Matching Under Limited Time". 

Link to paper: https://pubsonline.informs.org/doi/epdf/10.1287/opre.2022.2293

"matching_policies.py" containts the Python class "simulation" used to run the algorithms for the case study in the paper. This class supports the following the methods:

__init__(): Initializes the class
generate_input(): Calculates the cbar values for every type as explained in Appendix E.3 via the method initial_cbar() and the data files "ClusteredInput....feather" and "TypeCoordinate....feather". Then, prepares the time data for arrivals that will be used by the algorithms. Finally, calculates the minimum distance required to satisfy two rides in one route via the method MinDist().  
init_simulation(): Calculates departure times due to abandonment by generating exponentially distributed waiting times. 
MinDist(): Uses the data "TypePairDistance....csv" that contains the distance between every type pair to create the "attributes_graph....txt" and the "edges_graph....txt". Then creates the networkx graph that will be used in relation with gurobipy to run the algorithms.  
initial_cbar(): Uses the data file "ClusteredInput....feather" to derive agent types. Using the derived agent types, calculatez the cbar values for every type via the method cbar_multiple_node(). Finally, writes the results to the file "selfcbar_from_..._trip_neutral.txt".

The following methods are used for fast computation of the cbar values that are used in initial_cbar(). We use Numba, a JIT (just-in-time) compiler. It takes Python functions designated by particular annotations, and transforms as much as it can — via the LLVM (Low Level Virtual Machine) compiler — to efficient CPU and GPU (via CUDA for Nvidia GPUs and HSA for AMD GPUs) code:

cbar_single_node(): Calculate the cbar value for a particular node.
cbar_multiple_node(): Uses cbar_single_node() to calculate cbar values for multiple nodes. 
