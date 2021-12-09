import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies
from static.nav import navbar

card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Welcome", className="card-title"),
                html.P("Welcome to Bridge, the first online investment management service that bridges the gap between Main Street and Bay Street",
                    className="card-text",
                ),
            ], className='align-self-center'
        ),
    ],
    style={"width": "4in", "display": "inline-block"},
)

welcome_page = html.Div([

    html.Div([
        html.Img(src='/static/BridgeLogo.svg', style={'height':'2in', 'width':'4in'}),
    ], style={'align': 'center', 'marginTop': 50}
    ),

    html.Div([
        card
    ], style={'align': 'center'}
    ),

    html.Div([
        html.B("Let's Get Started", style={'color': 'black'}, className="me-1")
    ], style = {'marginBottom': 20, 'marginTop': 20}
    ),

    html.Div([
        dbc.Button("Log In", id='show-secret', outline=True, color="primary", className="me-1"),
        dbc.Button("Sign Up", color="primary", className="me-1", href='/sign_up'),
        html.Div(id='body-div')
    ],
        style={'marginBottom': 20, 'marginTop': 20}
    ),
],
    #className="d-grid gap-0 col-2 mx-auto",
    style={'textAlign': 'center'}
)

html.Div(id='secret-div')
