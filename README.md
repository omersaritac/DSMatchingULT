## Matching Policies
The code includes Python code for the OR paper "Dynamic Stochastic Matching Under Limited Time". 

Link to paper: https://pubsonline.informs.org/doi/epdf/10.1287/opre.2022.2293.

"matching_policies.py" contains the Python class "simulation" used to run the algorithms for the case study in the paper. This class supports the following methods that are used to create the objects to be used by the matching algorithms:

 * \_\_init\_\_(): Initializes the class "simulation".
 * generate_input(): Calculates the cbar values for every type as explained in Appendix E.3 via the method initial_cbar() and the data files.
 "ClusteredInput....feather" and "TypeCoordinate....feather". Then, prepares the time data for arrivals that will be used by the algorithms. Finally, calculates the minimum distance required to satisfy two rides in one route via the method MinDist().  
 * init_simulation(): Calculates departure times due to abandonment by generating exponentially distributed waiting times. 
 * MinDist(): Uses the data "TypePairDistance....csv" that contains the distance between every type pair to create the "attributes_graph....txt" and the "edges_graph....txt". Then creates the networkx graph that will be used in relation with gurobipy to run the algorithms.  
 * initial_cbar(): Uses the data file "ClusteredInput....feather" to derive agent types. Using the derived agent types, calculates the cbar values for every type via the method cbar_multiple_node(). Finally, writes the results to the file "selfcbar_from_..._trip_neutral.txt".

The following methods are used for fast computation of the cbar values that are used in initial_cbar(). We use Numba, a JIT (just-in-time) compiler. It takes Python functions designated by particular annotations, and transforms as much as it can — via the LLVM (Low Level Virtual Machine) compiler — to efficient CPU and GPU (via CUDA for Nvidia GPUs and HSA for AMD GPUs) code:

 * cbar_single_node(): Calculate the cbar value for a particular node.
 * cbar_multiple_node(): Uses cbar_single_node() to calculate cbar values for multiple nodes. 

The following methods of the class "simulation" run the matching algorithms:

 * threshold_policy(): Runs the threshold policy in the paper and writes the results to a csv file "ResultFrame_Threshold....csv".
 * Gurobi_LP_policy(): Runs the LP policy in the paper using the Gurobi Solver and writes the results to a csv file "ResultFrame_Gurobi_LP....csv".
 * Gurobi_batching_policy(): Runs the batching policy using the Gurobi Solver and writes the results to a csv file "ResultFrame_Gurobi_Batching....csv".

## Run File 

The code "RunFile.py" includes a sample code to run the algorithms in the case study of the paper.

##Dependencies



