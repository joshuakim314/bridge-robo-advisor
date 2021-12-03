import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies

ret = html.Div(
    [
        dbc.Label("Target Returns", html_for="slider", id='ret'),
        dcc.Slider(id="user-return", min=0, max=100, step=1, value=20, tooltip={"placement": "bottom", "always_visible": True}),
    ],
    className="mb-3",
)

risk = html.Div(
    [
        dbc.Label("Risk Tolerance", html_for="slider", id='risk'),
        dcc.Slider(id="user-risk", min=0, max=100, step=1, value=70, tooltip={"placement": "bottom", "always_visible": True}),
    ],
    className="mb-3",
)

control = html.Div(
    [
        dbc.Label("User Portfolio Control", html_for="slider", id='contr'),
        dcc.Slider(id="user-control", min=0, max=100, step=1, value=50, tooltip={"placement": "bottom", "always_visible": True}),

    ],
    className="mb-3",
)

card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Portfolio Parameters", className="card-title"),
                html.P("You may wish to alter these parameters and can do so at any time. We recommend leaving the default values to begin with and playing around later to see impacts on portfolio composition, expected returns, and expected variance.",
                    className="card-text",
                ),
            ], className='align-self-center'
        ),
    ],
    style={"width": "4in", "display": "inline-block"},
)

form = html.Div([
        html.Div(
            [
                dbc.Progress(id="progress", value=66, animated=True, striped=True),
            ],
            style={'marginBottom': '0.2in'}
        ),
        card,
        dbc.Form([ret, risk, control]),
        html.Div([
            dbc.Button("Back", color="primary", outline=True, href='/', className="me-3"),
            dbc.Button("Continue", id='continue-2', color="primary", className="me-3"),
        ],
        )
], style={"width": "4in", "display": "inline-block", 'marginTop': '0.2in'}
)

sign_up_2 = html.Div([
    html.Div([
        html.Img(src='/assets/BridgeLogo.svg', style={'height':'2in', 'width':'4in'}),
    ], style={'align': 'center', 'marginTop': 50}
    ),
    form,
    html.P(id='invalid', style={'color': 'red', 'marginTop': '0.2in'}),
    dbc.Tooltip(
            "Target returns of the portfolio. Note if target returns exceed feasible solutions, the maximum feasible target return will be used instead. We recommend values less than 20.",
            target="ret",
        ),
    dbc.Tooltip(
            "Higher risk tolerance means willing to trade higher variance for higher expected returns. We recommend values between 60 and 80",
            target="risk",
        ),
    dbc.Tooltip(
            "This parameter sets the minimum percentage of your portfolio that will be composed of your own stock picks, with the remaining being hedges decided by our portfolio optimization techniques.",
            target="contr",
        ),
], style={'textAlign': 'center'}
)