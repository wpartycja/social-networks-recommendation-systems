class ChainGraph:
    def __init__(self, num_nodes, directed=False, weighted=True):
        """
        Initializes the chain graph.

        Parameters:
        num_nodes (int): Number of nodes in the graph.
        directed (bool): Whether the graph is directed or not.
        weighted (bool): Whether the graph has weights on edges or not.
        """
        self.num_nodes = num_nodes
        self.directed = directed
        self.weighted = weighted
        self.edges = []

        for node_num in range(num_nodes - 1):
            self.add_edge(node_num, node_num + 1)

    def add_edge(self, source_node, dest_node, weight=1):
        """
        Adds an edge between nodes with an optional weight.

        Parameters:
        source_node (int): The source node.
        dest_node (int): The destination node.
        weight (float): The weight of the edge. Default is 1.
        """
        self.edges.append((source_node, dest_node, weight))

    def visualize(self):
        """
        Visualizes the chain graph.
        """
        graph_str = ""
        
        # Loop through each node and append node and edge in a single line
        for node in range(self.num_nodes - 1):
            source_node, dest_node = node, node + 1
            weight = next((w for x, y, w in self.edges if x == source_node and y == dest_node), 1)

            if self.directed:
                if self.weighted:
                    graph_str += f"({source_node}) --[{weight}]--> "
                else:
                    graph_str += f"({source_node}) -----> "
            else:
                if self.weighted:
                    graph_str += f"({source_node}) --[{weight}]-- "
                else:
                    graph_str += f"({source_node}) ------ "

        # last node
        graph_str += f"({self.num_nodes - 1})"

        print(graph_str)


# Example usage
if __name__ == "__main__":
    num_nodes = 6
    directed = False 
    weighted = False  

    chain_graph = ChainGraph(num_nodes, directed, weighted)

    # modification
    chain_graph.add_edge(2, 3, weight=5)

    chain_graph.visualize()
