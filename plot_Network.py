from graphviz import Digraph

dot = Digraph(comment='Dueling DQN Simplified')

# Define nodes
dot.node('Input', 'Input (10)')
dot.node('Feature', 'Feature Extraction\nLinear(10→256)\nReLU')
dot.node('Shared', 'Shared Features (256)')

dot.node('Value1', 'Value Stream\nLinear(256→128)')
dot.node('Value2', 'ReLU')
dot.node('Value3', 'Linear(128→1)')

dot.node('Adv1', 'Advantage Stream\nLinear(256→128)')
dot.node('Adv2', 'ReLU')
dot.node('Adv3', 'Linear(128→4)')

dot.node('Q', 'Q Value Calculation')

# Connect nodes
dot.edge('Input', 'Feature')
dot.edge('Feature', 'Shared')

# Value Stream Connections
dot.edge('Shared', 'Value1')
dot.edge('Value1', 'Value2')
dot.edge('Value2', 'Value3')

# Advantage Stream Connections
dot.edge('Shared', 'Adv1')
dot.edge('Adv1', 'Adv2')
dot.edge('Adv2', 'Adv3')

# Merge to Q Values
dot.edge('Value3', 'Q')
dot.edge('Adv3', 'Q')

# Generate the graph
dot.render('./images/network_view', format='png', cleanup=True)
