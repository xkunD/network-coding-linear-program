import networkx as nx
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class MaxConcurrentFlowSolver:
    """
    Finds the maximum flow that all sessions can achieve simultaneously.

    Solving Fractional Multi Commodity Flow is Polynomial Time.
    However, the print_solution method itself is NP cuz simple path finding.
    
    Solving Integral Multi Commodity Flow is NP hard.
    """
    
    def __init__(self, G: nx.Graph, sessions: List[Tuple[int, int]]):
        """
        Initialize the solver with a network and sessions.
        
        Args:
            G: NetworkX undirected graph with edge capacities stored as 'capacity' attribute
            sessions: List of (source, target) tuples representing commodity flows
        """
        self.G = G.copy()
        self.sessions = sessions
        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()
        self.num_commodities = len(sessions)
        
        # Store edges in a consistent order
        self.edges = list(G.edges())
        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        
        # Add reverse edges to the edge index for undirected graph
        for (u, v) in list(self.edge_index.keys()):
            self.edge_index[(v, u)] = self.edge_index[(u, v)]
        
        # Extract edge capacities
        self.capacities = np.array([G[u][v].get('capacity', float('inf')) for u, v in self.edges])
        
        # Create a directed graph for tracking flows (to handle edge directions properly)
        self.flow_graph = nx.DiGraph()
        for u, v in self.edges:
            capacity = G[u][v].get('capacity', float('inf'))
            self.flow_graph.add_edge(u, v, capacity=capacity)
            self.flow_graph.add_edge(v, u, capacity=capacity)
        
        # Initialize results
        self.flows = None
        self.max_concurrent_flow = None
        self.status = None
    
    def solve(self):
        """
        Solve the maximum concurrent flow problem using linear programming.
        
        Returns:
            Dict containing solution status and optimal flows
        """
        # Create flow variables for each commodity on each edge
        # f[k][i][j] represents flow of commodity k on edge (i,j)
        f = {}
        for k in range(self.num_commodities):
            f[k] = {}
            for i in range(self.num_nodes):
                f[k][i] = {}
                for j in self.G.neighbors(i):
                    f[k][i][j] = cp.Variable(nonneg=True)
        
        # Create variable for the maximum concurrent flow
        # This represents how much flow each session can carry
        mcf = cp.Variable(nonneg=True)
        
        # Objective: Maximize the concurrent flow
        objective = cp.Maximize(mcf)
        
        # Constraints
        constraints = []
        
        # 1. Capacity constraints for each edge (sum of all commodity flows <= edge capacity)
        for u, v in self.edges:
            edge_flow_sum = 0
            for k in range(self.num_commodities):
                edge_flow_sum += f[k][u][v] + f[k][v][u]  # Sum both directions for undirected
            constraints.append(edge_flow_sum <= self.G[u][v].get('capacity', float('inf')))
        
        # 2. Flow conservation constraints
        for k, (source, target) in enumerate(self.sessions):
            for i in range(self.num_nodes):
                if i != source and i != target:  # For intermediate nodes
                    # Sum of incoming flows equals sum of outgoing flows
                    flow_balance = 0
                    for j in self.G.neighbors(i):
                        flow_balance += f[k][j][i] - f[k][i][j]
                    constraints.append(flow_balance == 0)
        
        # 3. Flow requirements - each session must achieve the mcf value
        for k, (source, target) in enumerate(self.sessions):
            # Calculate the net outflow at source
            source_outflow = 0
            for j in self.G.neighbors(source):
                source_outflow += f[k][source][j] - f[k][j][source]
            constraints.append(source_outflow == mcf)
            
            # The net inflow at target should equal the outflow at source
            target_inflow = 0
            for i in self.G.neighbors(target):
                target_inflow += f[k][i][target] - f[k][target][i]
            constraints.append(target_inflow == mcf)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        # Store results
        self.status = problem.status
        self.max_concurrent_flow = mcf.value
        
        # Extract flow values for each commodity on each edge
        self.flows = {}
        for k in range(self.num_commodities):
            self.flows[k] = {}
            for u in range(self.num_nodes):
                for v in self.G.neighbors(u):
                    if (u, v) not in self.flows[k]:
                        # Get the flow values in both directions
                        forward_flow = f[k][u][v].value
                        backward_flow = f[k][v][u].value if v in f[k] and u in f[k][v] else 0
                        
                        # Calculate effective flow (can't have flow in both directions simultaneously)
                        if forward_flow > backward_flow:
                            self.flows[k][(u, v)] = forward_flow - backward_flow
                            self.flows[k][(v, u)] = 0
                        else:
                            self.flows[k][(v, u)] = backward_flow - forward_flow
                            self.flows[k][(u, v)] = 0
        
        return {
            'status': self.status,
            'max_concurrent_flow': self.max_concurrent_flow,
            'flows': self.flows
        }
    
    def get_flow_dict(self) -> Dict:
        """
        Returns a dictionary of flows for each session and edge.
        
        Returns:
            Dict: {session_idx: {(u, v): flow_value}}
        """
        if self.flows is None:
            raise ValueError("Problem must be solved before getting flows")
        return self.flows
    
    def visualize_network(self, figsize=(12, 8)):
        """
        Visualize the network with edge capacities and optimal flows.
        """
        if self.flows is None:
            raise ValueError("Problem must be solved before visualization")
        
        plt.figure(figsize=figsize)
        
        # Create position layout
        pos = nx.spring_layout(self.G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=500)
        
        # Draw edges with capacity labels
        edge_labels = {(u, v): f"cap: {self.G[u][v].get('capacity', '∞')}" for u, v in self.edges}
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        
        # Draw node labels
        nx.draw_networkx_labels(self.G, pos)
        
        # Add source and target information
        for k, (source, target) in enumerate(self.sessions):
            plt.annotate(f"Session {k}: {source}→{target} (flow: {self.max_concurrent_flow:.4f})",
                        xy=(0, 0), xytext=(0, -30 - 10 * k),
                        xycoords=('axes fraction'), textcoords='offset points',
                        ha='left', va='top')
        
        plt.title(f"Maximum Concurrent Flow: {self.max_concurrent_flow:.4f}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def print_solution(self):
        """
        Print the solution details.
        """
        if self.flows is None:
            raise ValueError("Problem must be solved before printing solution")
        
        print(f"Solution status: {self.status}")
        print(f"Maximum concurrent flow: {self.max_concurrent_flow:.4f}")
        
        for k, (source, target) in enumerate(self.sessions):
            print(f"\nSession {k}: {source} → {target} (flow: {self.max_concurrent_flow:.4f})")
            
            # Create a directed flow network for this commodity
            flow_net = nx.DiGraph()
            for u, v in self.G.edges():
                flow_uv = self.flows[k].get((u, v), 0)
                flow_vu = self.flows[k].get((v, u), 0)
                
                if flow_uv > 1e-6:
                    flow_net.add_edge(u, v, flow=flow_uv)
                if flow_vu > 1e-6:
                    flow_net.add_edge(v, u, flow=flow_vu)
            
            # Find all simple paths from source to target in the flow network
            try:
                all_paths = list(nx.all_simple_paths(flow_net, source, target))
                
                # Calculate flow for each path
                path_flows = []
                remaining_flow = self.max_concurrent_flow
                
                for path in all_paths:
                    if remaining_flow < 1e-6:
                        break
                        
                    # Find the minimum flow on the path
                    min_flow = float('inf')
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        edge_flow = flow_net[u][v].get('flow', 0)
                        min_flow = min(min_flow, edge_flow)
                    
                    # Skip paths with no flow
                    if min_flow < 1e-6:
                        continue
                    
                    # Adjust for remaining flow
                    path_flow = min(min_flow, remaining_flow)
                    path_flows.append((path, path_flow))
                    remaining_flow -= path_flow
                    
                    # Reduce the flow on this path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        flow_net[u][v]['flow'] -= path_flow
                
                # Print paths
                total_session_flow = 0
                for path, flow in path_flows:
                    print(f"  Path: {' → '.join(map(str, path))}, Flow: {flow:.4f}")
                    total_session_flow += flow
                
                print(f"  Total session flow: {total_session_flow:.4f}")
                
            except nx.NetworkXNoPath:
                print(f"  No flow path found from {source} to {target}")
                print(f"  Total session flow: 0.0000")
    
    def visualize_flows(self, figsize=(15, 10)):
        """
        Visualize the network with flows for each session.
        """
        if self.flows is None:
            raise ValueError("Problem must be solved before visualization")
        
        # Create a separate visualization for each session
        for k, (source, target) in enumerate(self.sessions):
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(self.G, seed=42)
            
            # Create a directed graph for this commodity's flow
            flow_graph = nx.DiGraph()
            flow_graph.add_nodes_from(self.G.nodes())
            
            # Add edges with flow values
            edge_colors = []
            edge_widths = []
            
            for u, v in self.G.edges():
                flow_uv = self.flows[k].get((u, v), 0)
                flow_vu = self.flows[k].get((v, u), 0)
                
                if flow_uv > 1e-6:
                    flow_graph.add_edge(u, v, weight=flow_uv)
                    edge_colors.append(flow_uv)
                    edge_widths.append(1 + 5 * flow_uv / self.max_concurrent_flow if self.max_concurrent_flow > 0 else 1)
                
                if flow_vu > 1e-6:
                    flow_graph.add_edge(v, u, weight=flow_vu)
                    edge_colors.append(flow_vu)
                    edge_widths.append(1 + 5 * flow_vu / self.max_concurrent_flow if self.max_concurrent_flow > 0 else 1)
            
            # Draw the nodes
            nx.draw_networkx_nodes(flow_graph, pos, node_size=700,
                                  node_color=['red' if n == source else 'green' if n == target else 'lightblue' 
                                             for n in flow_graph.nodes()])
            
            # Draw the edges with flow
            if flow_graph.edges():
                edges = nx.draw_networkx_edges(flow_graph, pos, width=edge_widths, edge_color=edge_colors, 
                                             edge_cmap=plt.cm.Blues, edge_vmin=0, edge_vmax=self.max_concurrent_flow,
                                             connectionstyle='arc3,rad=0.1')  # Curved edges for directed
                
                # Add a colorbar
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0, self.max_concurrent_flow))
                sm.set_array([])
                cbar = plt.colorbar(sm)
                cbar.set_label('Flow Amount')
                
                # Add edge labels with flow values
                edge_labels = {(u, v): f"{flow_graph[u][v]['weight']:.2f}" for u, v in flow_graph.edges()}
                nx.draw_networkx_edge_labels(flow_graph, pos, edge_labels=edge_labels, label_pos=0.3)
            
            # Draw node labels
            nx.draw_networkx_labels(flow_graph, pos)
            
            # Set title
            plt.title(f"Session {k}: {source} → {target} (Flow: {self.max_concurrent_flow:.4f})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()


# Example usage
def run_example():
    # Create a sample network
    G = nx.Graph()
    
    # Add nodes and edges with capacities
    edges = [
        (0, 3, 1),
        (0, 4, 1),
        (3, 1, 1),
        (1, 4, 1),
        (3, 2, 1),
        (2, 4, 1)
    ]
    for u, v, capacity in edges:
        G.add_edge(u, v, capacity=capacity)
    
    # Define sessions: (source, target)
    sessions = [
        (0, 2),
        (1, 0),
        (2, 1),
        (3, 4)
    ]
    
    # Create and solve the maximum concurrent flow problem
    solver = MaxConcurrentFlowSolver(G, sessions)
    result = solver.solve()
    
    # Display results
    solver.print_solution()
    # solver.visualize_network()
    # solver.visualize_flows()


if __name__ == "__main__":
    run_example()