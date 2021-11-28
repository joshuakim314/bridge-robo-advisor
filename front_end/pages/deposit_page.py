import dash_bootstrap_components as dbc
from dash import Dash, State, html, dcc, dash_table, callback, dependencies
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform

from assets.nav import navbar
from assets.examples import exgraph1, exgraph2, extable1, extable2

deposit_page = html.Div([
    navbar,
    html.H2(id='test-callback', style={'marginTop': '0.2in', 'marginLeft': '1.2in', 'marginRight': '1.2in', 'textAlign': 'center'}),
    html.Div([
            dbc.Button("Deposit", id='deposit-show-secret', color="primary", outline=False, className="me-3", size='lg'),
            dbc.Button("Withdraw", id='withdraw-show-secret', color="danger", outline=True, className="me-3", size='lg'),
            html.Div(id='deposit-secret-div'),
            html.Div(id='withdraw-secret-div')
        ], style={'marginTop': '0.2in', 'marginLeft': '1.2in', 'marginRight': '1.2in', 'textAlign': 'center'}
    ),


],
    #className="d-grid gap-0 col-2 mx-auto",

)

