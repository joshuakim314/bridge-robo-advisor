import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies
from static.nav import navbar

nyi_page = html.Div([
    navbar,
    html.Div([
        html.Img(src='/assets/BridgeLogo.svg', style={'height':'2in', 'width':'4in'}),
    ], style={'align': 'center', 'marginTop': 50}
    ),
    html.H1('Sorry'),
    html.H2('This page has not yet been implemented')
], style={'textAlign': 'center'}
)