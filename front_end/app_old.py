import dash_bootstrap_components as dbc
from dash import State, html, dcc, dash_table, callback, dependencies
from dash_extensions.enrich import Dash, Output, DashProxy, Input, MultiplexerTransform

#app = DashProxy(transforms=[MultiplexerTransform()], external_stylesheets=[dbc.themes.MINTY], suppress_callback_exceptions=True)
app = DashProxy(name=__name__,
            title="My App",
            assets_folder="static",
            assets_url_path="static",
            transforms=[MultiplexerTransform()],
            external_stylesheets=[dbc.themes.MINTY],
            suppress_callback_exceptions=True
           )
server = app.server