import dash_bootstrap_components as dbc
from dash import Dash, State, html, dcc, dash_table, callback, dependencies
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform

app = DashProxy(transforms=[MultiplexerTransform()], external_stylesheets=[dbc.themes.MINTY], suppress_callback_exceptions=True)
server = app.server