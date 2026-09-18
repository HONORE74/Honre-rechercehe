from dash import Dash, html
app = Dash(__name__)
app.layout = html.Div("Ca marche.")
app.run(jupyter_mode="inline", port=8060)
