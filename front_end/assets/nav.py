import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies

logo = '/assets/BridgeLogo.svg'

buttons = html.Div([
    dbc.NavItem(dbc.NavLink("Portfolio", href='/main')),
    dbc.NavItem(dbc.NavLink("Deposits", href='/deposit')),
    dbc.NavItem(dbc.NavLink("Breakdown", href='/nyi')),
    dbc.NavItem(dbc.NavLink("Preferences", href='/prefs')),
    dbc.Button("Log Out", outline=False, color="primary", href='/', id='log-out', n_clicks=0),
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
                href='main',
                style={"textDecoration": "none"},
            ),
            buttons,
        ]
    ),
    color="light",
    dark=True,
)