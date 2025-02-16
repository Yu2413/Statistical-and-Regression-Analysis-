from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Random Forest', format='png')

# Define the nodes for the Random Forest
dot.node('RF', 'Random Forest', shape='ellipse', style='filled', color='lightblue')

# Add individual decision trees to the forest
dot.node('T1', 'Tree 1', shape='box', style='filled', color='lightgreen')
dot.node('T2', 'Tree 2', shape='box', style='filled', color='lightgreen')
dot.node('T3', 'Tree 3', shape='box', style='filled', color='lightgreen')
dot.node('T4', 'Tree 4', shape='box', style='filled', color='lightgreen')

# Connect the trees to the Random Forest node
dot.edge('RF', 'T1', label='Subset of Features')
dot.edge('RF', 'T2', label='Subset of Features')
dot.edge('RF', 'T3', label='Subset of Features')
dot.edge('RF', 'T4', label='Subset of Features')

# Add nodes for the final aggregation (e.g., majority voting or averaging)
dot.node('Agg', 'Aggregation\n(Majority Voting / Averaging)', shape='ellipse', style='filled', color='lightcoral')

# Connect the trees to the aggregation node
dot.edge('T1', 'Agg', label='Prediction')
dot.edge('T2', 'Agg', label='Prediction')
dot.edge('T3', 'Agg', label='Prediction')
dot.edge('T4', 'Agg', label='Prediction')

# Save and render the diagram to a file and open it
dot.render('random_forest_model', view=True, cleanup=True)