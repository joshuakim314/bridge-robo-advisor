import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies

risk = html.Div(
    [
        dbc.Label("Risk Parameter", html_for="slider"),
        dcc.Slider(id="user-risk", min=0, max=10, step=1, value=3, tooltip={"placement": "bottom", "always_visible": True}),
    ],
    className="mb-3",
)

control = html.Div(
    [
        dbc.Label("User Portfolio Control", html_for="slider"),
        dcc.Slider(id="user-control", min=0, max=10, step=1, value=3, tooltip={"placement": "bottom", "always_visible": True}),

    ],
    className="mb-3",
)

form = html.Div([
        html.Div(
            [
                dbc.Progress(id="progress", value=66, animated=True, striped=True),
            ],
            style={'marginBottom': '0.2in'}
        ),
        dbc.Form([risk, control]),
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
    html.P(id='invalid', style={'color': 'red', 'marginTop': '0.2in'})
], style={'textAlign': 'center'}
)