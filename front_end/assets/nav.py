import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies

logo = '/assets/BridgeLogo.svg'

buttons = html.Div([
    dbc.Button("Left", outline=False, color="primary"),
    dbc.Button("Right", outline=True, color="primary"),
    dbc.Button("Right", outline=True, color="primary"),
    dbc.Button("Right", outline=True, color="primary"),
],

    className="d-grid gap-2 d-md-flex justify-content-md-end",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=logo, height="30px")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href='/test',
                style={"textDecoration": "none"},
            ),
            buttons,
        ]
    ),
    color="light",
    dark=True,
)