import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies

email_input = html.Div(
    [
        dbc.Input(type="email", id="example-email", placeholder="Enter email"),
    ],
    className="mb-3",
)

confirm_email = html.Div(
    [
        dbc.Input(type="email", id="example-email", placeholder="Confirm email"),
    ],
    className="mb-3",
)

password_input = html.Div(
    [
        dbc.Input(
            type="password",
            id="example-password",
            placeholder="Enter password",
        ),
    ],
    className="mb-3",
)

confirm_password = html.Div(
    [
        dbc.Input(
            type="password",
            id="example-password",
            placeholder="Confirm password",
        ),
    ],
    className="mb-3",
)

inline_name = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Input(
                    type="name",
                    id="first-name",
                    placeholder="First Name",
                ),
            ],
            width=6,
        ),
        dbc.Col(
            [
                dbc.Input(
                    type="name",
                    id="example-name-grid",
                    placeholder="Last Name",
                ),
            ],
            width=6,
        ),
    ],
    className="mb-3",
)

form = html.Div([
        dbc.Form([inline_name, email_input, confirm_email, password_input, confirm_password]),
        html.Div([
            dbc.Button("Continue", color="primary", href='/main', className="me-3"),
            dbc.Button("Back", color="primary", outline=True, href='/', className="me-3"),
        ],
        )
], style={"width": "4in", "display": "inline-block", 'marginTop': '0.2in'}
)

sign_up = html.Div([
    html.Div([
        html.Img(src='/assets/BridgeLogo.svg', style={'height':'2in', 'width':'4in'}),
    ], style={'align': 'center', 'marginTop': 50}
    ),
    form
], style={'textAlign': 'center'}
)