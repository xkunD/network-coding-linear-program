import networkx as nx
import itertools
from pulp import *
import matplotlib.pyplot as plt
from collections import defaultdict # Added import

class EntropyCalculusAnalyzer:
    """
    A general script for analyzing network capacity using entropy calculus.
    """
    def __init__(self):
        self.G = None  # Undirected graph
        self.DG = None  # Directed graph
        self.source_sink_pairs = []
        self.entropy_vars = {}  # Dictionary to store all entropy variables
        self.results = {}

    def create_network(self, edges, capacities=None):
        self.G = nx.Graph()
        self.G.add_edges_from(edges)
        if capacities:
            for edge, capacity in capacities.items():
                u, v = edge
                self.G[u][v]['capacity'] = capacity
        else:
            nx.set_edge_attributes(self.G, 1, 'capacity')
        self.DG = self._derive_directed_graph()
        return self

    def set_source_sink_pairs(self, pairs):
        self.source_sink_pairs = pairs
         # Store mapping for quick lookup
        self.source_map = {s: i for i, (s, t) in enumerate(pairs)}
        self.sink_map = {t: i for i, (s, t) in enumerate(pairs)}
        return self

    def _derive_directed_graph(self):
        DG = nx.DiGraph()
        DG.add_nodes_from(self.G.nodes)
        for u, v, data in self.G.edges(data=True):
            capacity = data.get('capacity', 1)
            DG.add_edge(u, v, capacity=capacity)
            DG.add_edge(v, u, capacity=capacity)
        return DG

    def get_all_cuts(self):
        nodes = list(self.G.nodes)
        cuts = []
        # Limit combinations for very large graphs if needed
        max_nodes_for_all_cuts = 10 
        num_nodes = len(nodes)
        if num_nodes > max_nodes_for_all_cuts:
             print(f"Warning: Graph has {num_nodes} nodes. Calculating all cuts may be slow.")
             # Implement sampling or heuristics if needed for large graphs
        
        # Generate all non-empty proper subsets
        for i in range(1, 1 << num_nodes):
             S = set()
             for j in range(num_nodes):
                  if (i >> j) & 1:
                       S.add(nodes[j])
             if S and len(S) < num_nodes: # Ensure non-empty proper subset
                S_complement = set(nodes) - S
                cuts.append((S, S_complement))
        # Remove duplicate partitions (e.g., ({a}, {b,c}) and ({b,c}, {a}))
        unique_cuts = set()
        final_cuts = []
        for s_set, sc_set in cuts:
            s_tuple = tuple(sorted(list(s_set)))
            sc_tuple = tuple(sorted(list(sc_set)))
            # Canonical representation (lexicographically smaller first)
            cut_repr = tuple(sorted((s_tuple, sc_tuple)))
            if cut_repr not in unique_cuts:
                unique_cuts.add(cut_repr)
                final_cuts.append((s_set, sc_set))

        return final_cuts


    def get_incoming_edges(self, S):
        """Get incoming graph edges to a set of nodes S."""
        incoming = []
        for v in S:
            for u in self.DG.predecessors(v):
                if u not in S:
                    incoming.append((u, v))
        return incoming

    def get_outgoing_edges(self, S):
        """Get outgoing graph edges from a set of nodes S."""
        outgoing = []
        for u in S:
            for v in self.DG.successors(u):
                if v not in S:
                    outgoing.append((u, v))
        return outgoing

    def formulate_lp(self):
        """
        Formulate a linear program for network capacity analysis.
        """
        prob = LpProblem("Network_Capacity", LpMaximize)

        # 1. Create variables for all sessions (source-sink pairs)
        for i, (source, sink) in enumerate(self.source_sink_pairs):
            var_name = f"X{i+1}"
            self.entropy_vars[var_name] = LpVariable(var_name, lowBound=0)

        # 2. Create variables for all directed edges
        for u, v in self.DG.edges():
            var_name = f"H_{u}_{v}"
            self.entropy_vars[var_name] = LpVariable(var_name, lowBound=0)

        # 3. Create variables for auxiliary cut entropies
        all_cuts = self.get_all_cuts()
        for S, S_complement in all_cuts:
            s_label = '_'.join(sorted(S))
            # Check if any inputs (edges or sources) or outputs (edges or sinks) exist for S
            has_inputs = bool(self.get_incoming_edges(S) or any(s in S for s,t in self.source_sink_pairs))
            has_outputs = bool(self.get_outgoing_edges(S) or any(t in S for s,t in self.source_sink_pairs))

            if has_inputs:
                self.entropy_vars[f"H_in_{s_label}"] = LpVariable(f"H_in_{s_label}", lowBound=0)
            if has_outputs:
                 self.entropy_vars[f"H_out_{s_label}"] = LpVariable(f"H_out_{s_label}", lowBound=0)
            # Need H_in_out only if both inputs and outputs exist conceptually for IO inequality
            if has_inputs and has_outputs:
                 self.entropy_vars[f"H_in_out_{s_label}"] = LpVariable(f"H_in_out_{s_label}", lowBound=0)


        # 4. Set objective: maximize H(X1) (or average rate)
        prob += self.entropy_vars["X1"]

        # 5. Add constraint: all sessions have the same entropy rate R
        session_rate = self.entropy_vars["X1"] # R
        for i in range(1, len(self.source_sink_pairs)):
            prob += self.entropy_vars[f"X{i+1}"] == session_rate

        # 6. Add capacity constraints for undirected edges
        for u, v in self.G.edges():
            # Ensure variables exist before adding constraint
            h_uv_name = f"H_{u}_{v}"
            h_vu_name = f"H_{v}_{u}"
            if h_uv_name in self.entropy_vars and h_vu_name in self.entropy_vars:
                 prob += (self.entropy_vars[h_uv_name] + self.entropy_vars[h_vu_name]
                         <= self.G[u][v].get('capacity', 1)), f"Capacity_{u}_{v}"

        # 7. Add Crypto Inequalities for all cuts
        # H(Sessions_crossing_cut | Edges_crossing_cut) = 0
        # Relaxed to: sum H(Xi crossing) <= H_cut <= sum H(Edges crossing)
        # print("Adding Crypto Constraints...") # Added print statement
        crypto_constraints_added = 0
        for S, S_complement in all_cuts: # Assuming all_cuts is available
            s_label = '_'.join(sorted(S))

            # DEM(S): Sessions crossing the cut
            dem_S_vars = []
            dem_indices = [] # Store indices for variable names if needed
            for i, (s, t) in enumerate(self.source_sink_pairs):
                if (s in S and t in S_complement) or (s in S_complement and t in S):
                    var_name = f"X{i+1}"
                    if var_name in self.entropy_vars:
                        dem_S_vars.append(self.entropy_vars[var_name])
                        dem_indices.append(i+1)

            # CUT(S): Edges crossing the cut (in both directions)
            cut_S_edge_vars = []
            cut_edge_names = [] # Store names for debugging if needed
            crossing_edges = []
            # Edges S -> S_complement
            for u in S:
                for v in self.DG.successors(u):
                    if v in S_complement:
                        crossing_edges.append((u,v))
            # Edges S_complement -> S
            for u in S_complement:
                for v in self.DG.successors(u):
                    if v in S:
                        crossing_edges.append((u,v))

            for u, v in crossing_edges:
                edge_name = f"H_{u}_{v}"
                if edge_name in self.entropy_vars:
                    cut_S_edge_vars.append(self.entropy_vars[edge_name])
                    cut_edge_names.append(edge_name)

            # Only add if there are sessions demanding transfer across the cut
            # and edges allowing transfer across the cut
            if dem_S_vars and cut_S_edge_vars:
                # Create auxiliary variable H_cut_S
                cut_var_name = f"H_cut_{s_label}"
                # Avoid recreating if it already exists (can happen with symmetric cuts)
                if cut_var_name not in self.entropy_vars:
                    H_cut_S = LpVariable(cut_var_name, lowBound=0)
                    self.entropy_vars[cut_var_name] = H_cut_S
                else:
                    H_cut_S = self.entropy_vars[cut_var_name]
                
                crypto_constraints_added += 1

                # Upper bound H_cut_S by sum of crossing edge entropies
                prob += H_cut_S <= lpSum(cut_S_edge_vars), f"CryptoUpperBound_{s_label}"

                # Monotonicity: Each crossing edge entropy <= H_cut_S
                for edge_var in cut_S_edge_vars:
                    prob += edge_var <= H_cut_S, f"CryptoMonoEdge_{edge_var.name}_vs_{s_label}"

                # Crypto Inequality Relaxation: sum H(Sessions crossing) <= H_cut_S
                prob += lpSum(dem_S_vars) <= H_cut_S, f"Crypto_{s_label}"

        # Optional: Print how many were added
        print(f"Added {crypto_constraints_added} Crypto Inequality constraint sets.")



        # 7. Add input-output and related monotonicity constraints for all cuts
        for S, S_complement in all_cuts:
            s_label = '_'.join(sorted(S))
            
            # Identify components for this cut
            in_edges = self.get_incoming_edges(S)
            out_edges = self.get_outgoing_edges(S)
            
            sources_in_S_idx = [i for i, (s, t) in enumerate(self.source_sink_pairs) if s in S]
            sinks_in_S_idx = [i for i, (s, t) in enumerate(self.source_sink_pairs) if t in S]

            sources_in_S_vars = [self.entropy_vars[f"X{i+1}"] for i in sources_in_S_idx]
            sinks_in_S_vars = [self.entropy_vars[f"X{i+1}"] for i in sinks_in_S_idx] # Used for H_out and H_in_out bounds

            in_edge_vars = [self.entropy_vars[f"H_{u}_{v}"] for u, v in in_edges]
            out_edge_vars = [self.entropy_vars[f"H_{u}_{v}"] for u, v in out_edges]

            # Define H_in_S and its relations (if it exists)
            in_S_name = f"H_in_{s_label}"
            if in_S_name in self.entropy_vars:
                H_in_S = self.entropy_vars[in_S_name]
                # Upper bound relaxation: H_in_S <= sum of components
                prob += H_in_S <= lpSum(in_edge_vars) + lpSum(sources_in_S_vars), f"UpperBound_{in_S_name}"
                # Monotonicity: Each component <= H_in_S
                for edge_var in in_edge_vars:
                    prob += edge_var <= H_in_S, f"MonoInEdge_{edge_var.name}_in_{s_label}"
                for source_var in sources_in_S_vars:
                    prob += source_var <= H_in_S, f"MonoInSource_{source_var.name}_in_{s_label}"

            # Define H_out_S and its relations (if it exists)
            out_S_name = f"H_out_{s_label}"
            if out_S_name in self.entropy_vars:
                H_out_S = self.entropy_vars[out_S_name]
                 # Upper bound relaxation: H_out_S <= sum of components (edges out + sinks in S)
                prob += H_out_S <= lpSum(out_edge_vars) + lpSum(sinks_in_S_vars), f"UpperBound_{out_S_name}"
                 # Monotonicity: Each component <= H_out_S
                for edge_var in out_edge_vars:
                     prob += edge_var <= H_out_S, f"MonoOutEdge_{edge_var.name}_out_{s_label}"
                for sink_var in sinks_in_S_vars: # X_i where t_i in S
                     prob += sink_var <= H_out_S, f"MonoOutSink_{sink_var.name}_out_{s_label}"


            # Define H_in_out_S and apply IO (if it exists)
            both_S_name = f"H_in_out_{s_label}"
            if both_S_name in self.entropy_vars:
                H_in_out_S = self.entropy_vars[both_S_name]
                
                # Upper bound relaxation: H_in_out_S <= sum of all components (in/out edges + sources/sinks in S)
                prob += H_in_out_S <= lpSum(in_edge_vars) + lpSum(out_edge_vars) + \
                                      lpSum(sources_in_S_vars) + lpSum(sinks_in_S_vars), f"UpperBound_{both_S_name}"

                # Monotonicity H_in <= H_in_out and H_out <= H_in_out
                if in_S_name in self.entropy_vars:
                    prob += self.entropy_vars[in_S_name] <= H_in_out_S, f"Mono_In_vs_InOut_{s_label}"
                if out_S_name in self.entropy_vars:
                     prob += self.entropy_vars[out_S_name] <= H_in_out_S, f"Mono_Out_vs_InOut_{s_label}"

                # Apply Input-Output Inequality Relaxation: H(in U out) <= H(in)
                # Use the variables representing the upper bounds
                if in_S_name in self.entropy_vars: # Check H_in_S exists
                    prob += H_in_out_S <= self.entropy_vars[in_S_name], f"IO_{s_label}"


        # 8. Add submodularity constraints (Placeholder - complex to add all)
        # Example: H(A) + H(B) >= H(A U B) + H(A n B)
        # This is partially captured by the monotonicity constraints added above.
        # Adding all submodularity constraints is usually infeasible.

        # 9. Add joint decodability constraint per sink (Relaxation: sum H(Xi) <= sum H(in_edges))
        sink_to_sessions = defaultdict(list)
        for i, (source, sink) in enumerate(self.source_sink_pairs):
            var_name = f"X{i+1}"
            sink_to_sessions[sink].append(self.entropy_vars[var_name])

        for sink, session_vars_at_sink in sink_to_sessions.items():
            in_edges_to_sink = list(self.DG.in_edges(sink))
            if in_edges_to_sink: # Only if sink has incoming edges
                in_edge_vars_to_sink = [self.entropy_vars[f"H_{u}_{v}"] for u, v in in_edges_to_sink]
                # Constraint: Sum(H(Xi) for sink) <= Sum(H(Y_uv) for incoming edges)
                prob += lpSum(session_vars_at_sink) <= lpSum(in_edge_vars_to_sink), f"JointDecodability_{sink}"

        # 10. Relate Session variables to Cut entropy (Monotonicity: Xi <= H(Cut))
        for i, (source, sink) in enumerate(self.source_sink_pairs):
            session_var = self.entropy_vars[f"X{i+1}"]

            for S, S_complement in all_cuts:
                if ((source in S and sink in S_complement) or
                    (source in S_complement and sink in S)):
                    # If session (i) crosses cut S, then H(Xi) <= H(delta_in(S) U delta_out(S))
                    # Use H_in_out_S as the upper bound variable
                    s_label = '_'.join(sorted(S))
                    both_S_name = f"H_in_out_{s_label}"
                    if both_S_name in self.entropy_vars: # Check variable exists
                         prob += session_var <= self.entropy_vars[both_S_name], f"SessionCut_{session_var.name}_vs_{s_label}"

        return prob

    def analyze_network(self):
        """
        Analyze the network by formulating and solving the LP.
        """
        prob = self.formulate_lp()
        
        # Optional: Write LP to file for debugging
        # prob.writeLP("network_capacity.lp") 
        
        print("Solving LP...")
        # Solve with CBC solver, enable messages for debugging if needed
        prob.solve(PULP_CBC_CMD(msg=False)) # Set msg=True to see solver output
        
        print(f"Solver status: {LpStatus[prob.status]}")
        
        self.results['status'] = LpStatus[prob.status]
        if prob.status == LpStatusOptimal:
            self.results['max_rate'] = value(prob.objective)
            var_values = {}
            for name, var in self.entropy_vars.items():
                var_values[name] = var.value()
            self.results['variables'] = var_values
        elif prob.status == LpStatusInfeasible:
             print("Error: The LP problem is infeasible. Check constraints.")
             self.results['max_rate'] = None
        else:
             print(f"Warning: Optimal solution not found. Status: {LpStatus[prob.status]}")
             try:
                  self.results['max_rate'] = value(prob.objective) # May be None or incorrect
             except:
                  self.results['max_rate'] = None


        return self.results

    def print_results(self):
        """Print the analysis results."""
        print("\n====== Network Capacity Analysis Results ======")
        print(f"Network: {len(self.G.nodes)} nodes, {len(self.G.edges())} edges")
        print(f"Source-sink pairs: {self.source_sink_pairs}")
        print(f"LP Status: {self.results.get('status', 'N/A')}")


        if 'max_rate' in self.results and self.results['max_rate'] is not None:
            print(f"\nMaximum achievable rate (Upper Bound): {self.results['max_rate']:.8f}")

            if 'variables' in self.results:
                print("\nSession Variable Values:")
                for i in range(len(self.source_sink_pairs)):
                     var_name = f"X{i+1}"
                     val = self.results['variables'].get(var_name, 'N/A')
                     print(f"H({var_name}) = {val if val != 'N/A' else val:.8f}")

                # Optional: Print edge variables or other details if needed for debugging
                print("\nExample Edge Variable Values:")
                count = 0
                for name, value in self.results['variables'].items():
                    if name.startswith("H_") and "_" in name and len(name.split('_')) == 3 and count < 20:
                        print(f"{name} = {value:.4f}")
                        count += 1
                print("\nExample Cut Variable Values:")
                count = 0
                for name, value in self.results['variables'].items():
                     if name.startswith("H_in_out") and count < 10:
                          print(f"{name} = {value:.4f}")
                          count += 1

        else:
             print("\nCould not determine maximum achievable rate.")

        print("\n===========================================")


# Example Usage
if __name__ == "__main__":
    analyzer = EntropyCalculusAnalyzer()

    edges = [
        ('a', 'd'), ('a', 'e'),
        ('b', 'd'), ('b', 'e'),
        ('c', 'd'), ('c', 'e')
    ]
    analyzer.create_network(edges)

    # Use the ACYCLIC configuration from the paper's Theorem 1
    pairs = [
        ('a', 'b'),  # X1
        ('b', 'c'),  # X2
        ('a', 'c'),  # X3
        ('d', 'e')   # X4
    ]
    # # Optional: Use the CYCLIC configuration from Theorem 3
    # pairs = [
    #     ('a', 'b'),  # X1
    #     ('b', 'c'),  # X2
    #     ('c', 'a'),  # X3  <- Changed
    #     ('d', 'e')   # X4
    # ]
    analyzer.set_source_sink_pairs(pairs)

    analyzer.analyze_network()
    analyzer.print_results()