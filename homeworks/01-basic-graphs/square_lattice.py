class SquareLatticeGraph:
    def __init__(self, rows, cols):
        """
        Initializes a square lattice graph.

        Parameters:
        rows (int): Number of rows in the lattice.
        cols (int): Number of columns in the lattice.
        """
        self.rows = rows
        self.cols = cols
        self.edges = []

        # Connect horizontal and vertical neighbors to form a lattice
        for row in range(rows):
            for col in range(cols):
                node = row * cols + col 

                if col < cols - 1:
                    self.add_edge(node, node + 1)

                if row < rows - 1:
                    self.add_edge(node, node + cols)

    def add_edge(self, source_node, dest_node):
        """
        Adds an undirected edge between nodes.
        """
        
        self.edges.append((source_node, dest_node))
    
    def compute_vertex_degrees(self):
        """
        Computes the degree of each vertex in the graph.
        """
        degree_count = [0] * (self.rows * self.cols) 
        
        for u, v in self.edges:
            degree_count[u] += 1  
            degree_count[v] += 1 
        
        print("\nVertex Degrees:")
        for i in range(len(degree_count)):
            print(f"Node {i:02}: Degree {degree_count[i]}")

    def visualize(self):
        """
        Visualizes the square lattice graph.
        """
        horizontal_edge = "-----"
        vertical_edge = " |  "

        for row in range(self.rows):
            # print horizontal
            node_row = ""
            for col in range(self.cols):
                node_index = row * self.cols + col
                node_row += f"({node_index:02})"  # Format the index as two digits
                if col < self.cols - 1:
                    node_row += horizontal_edge  # Horizontal edges between nodes
            print(node_row)

            # print vertical
            if row < self.rows - 1:
                edge_row = ""
                for col in range(self.cols):
                    edge_row += vertical_edge
                    if col < self.cols - 1:
                        edge_row += "     "  # Space between vertical edges
                print(edge_row)


# Example usage
if __name__ == "__main__":
    rows = 6
    cols = 2

    lattice_graph = SquareLatticeGraph(rows, cols)
    lattice_graph.visualize()
    lattice_graph.compute_vertex_degrees()