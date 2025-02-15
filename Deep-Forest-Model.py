from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Deep Forest Model', format='png')

# Define the nodes
dot.node('A', 'Input Features')
dot.node('B', 'Multi-Grained Scanning')
dot.node('C', 'Cascade Layer 1\n(Random Forest, Extra Trees)')
dot.node('D', 'Cascade Layer 2\n(Random Forest, Extra Trees)')
dot.node('E', 'Cascade Layer 3\n(Random Forest, Extra Trees)')
dot.node('F', 'Prediction Output')

# Create edges between nodes to represent data flow
dot.edge('A', 'B', label='Feature Extraction')
dot.edge('B', 'C', label='Processed Features')
dot.edge('C', 'D', label='Processed Features')
dot.edge('D', 'E', label='Processed Features')
dot.edge('E', 'F', label='Final Decision')

# Save and render the diagram to a file and open it
dot.render('deep_forest_model', view=True, cleanup=True)
