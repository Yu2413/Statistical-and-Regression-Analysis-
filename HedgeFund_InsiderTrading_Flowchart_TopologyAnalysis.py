from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Relationship between Hedge Fund and Insider Investing')

# Define nodes
dot.node('HF', 'Hedge Fund Investing')
dot.node('II', 'Insider Investing\n(Buying/Selling by Company Insiders)')
dot.node('SI', 'Signal Extraction\n(Identifying Insider Activity)')
dot.node('MD', 'Market Decision\n(Investment Strategies)')
dot.node('MI', 'Market Impact\n(Price & Sentiment)')

# Define relationships (edges)
# Insider investing provides signals that hedge funds may use.
dot.edge('II', 'SI', label='Insider data as signal')
# Hedge funds extract signals from insider activity.
dot.edge('SI', 'HF', label='Inform strategies')
# Hedge funds execute strategies affecting the market.
dot.edge('HF', 'MI', label='Investments influence')
# Changes in market impact may, in turn, influence insider behavior.
dot.edge('MI', 'II', label='Feedback to insiders')
# Hedge funds also base decisions on market data.
dot.edge('MI', 'MD', label='Market data analysis')
dot.edge('MD', 'HF', label='Refine strategies')

# Render and display the flowchart
dot.render('hedge_fund_insider_flowchart.gv', view=True)
