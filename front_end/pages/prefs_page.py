import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies
from assets.nav import navbar

prefs_page = html.Div([
    navbar,
    html.Div([
        html.Img(src='/assets/BridgeLogo.svg', style={'height':'2in', 'width':'4in'}),
    ], style={'align': 'center', 'marginTop': 50}
    ),
    html.H1('Stock Preferences'),
    dbc.Button("Add Filter", id="add-filter", n_clicks=0),
    html.Div(id='dropdown-container', children=[], style={'align': 'center', 'marginTop': '0.2in'}),
    dbc.Button("Confirm Preferences", id="confirm-preferences", n_clicks=0, style={'align': 'center', 'marginTop': '0.2in'})
], style={'textAlign': 'center'}
)

#