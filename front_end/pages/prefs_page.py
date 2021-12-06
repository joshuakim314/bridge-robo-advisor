import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies
from assets.nav import navbar

ret = html.Div(
    [
        dbc.Label("Target Returns", html_for="slider", id='ret'),
        dcc.Slider(id="user-return-set", min=0, max=100, step=1, tooltip={"placement": "bottom", "always_visible": True},
                   marks={0: {'label': '0%', 'style': {'color': '#313131'}},
                          100: {'label': '100%', 'style': {'color': '#313131'}}
                          },),
    ],
    className="mb-3",
)

risk = html.Div(
    [
        dbc.Label("Risk Tolerance", html_for="slider", id='risk'),
        dcc.Slider(id="user-risk-set", min=0, max=100, step=1, tooltip={"placement": "bottom", "always_visible": True},
                   marks={0: {'label': '0%', 'style': {'color': '#313131'}},
                          100: {'label': '100%', 'style': {'color': '#313131'}}
                          },),
    ],
    className="mb-3",
)

control = html.Div(
    [
        dbc.Label("User Portfolio Control", html_for="slider", id='contr'),
        dcc.Slider(id="user-control-set", min=0, max=100, step=1, tooltip={"placement": "bottom", "always_visible": True},
                   marks={0: {'label': '0%', 'style': {'color': '#313131'}},
                          100: {'label': '100%', 'style': {'color': '#313131'}}
                          },
                   ),

    ],
    className="mb-3",
)

horizon = html.Div(
    [
        dbc.Label("User Investment Horizon", html_for="slider", id='horiz'),
        dcc.Slider(id="user-horizon-set", min=1, max=24, step=1, tooltip={"placement": "bottom", "always_visible": True},
                   marks={1: {'label': '1 Month', 'style': {'color': '#313131'}},
                          24: {'label': '24 Months', 'style': {'color': '#313131'}}
                          },
                   ),

    ],
    className="mb-3",
)

max_etfs = html.Div(
    [
        dbc.Label("Hedging Cardinality", html_for="slider", id='max-card'),
        dcc.Slider(id="user-card-set", min=0, max=50, step=1, tooltip={"placement": "bottom", "always_visible": True},
                   marks={0: {'label': '0 ETFs', 'style': {'color': '#313131'}},
                          50: {'label': '50 ETFs', 'style': {'color': '#313131'}}
                          },
                   ),

    ],
    className="mb-3",
)

prefs_page = html.Div([
    navbar,
    html.Div(
        id='show-hype-graph',
        style={'marginTop': '0.2in', 'marginLeft': '0.5in', 'marginRight': '0.5in'}

    ),
    dbc.Row([
        dbc.Col(
            [
                html.H1('Portfolio Parameters'),
                html.Div([
                    dbc.Form([ret, risk, control, horizon, max_etfs]),
                ], style={'marginRight': '0.2in'}
                )
            ]
        ),
        dbc.Col(
            [
                html.H1('Stock Preferences'),
                dbc.Button("Add Filter", id="add-filter", n_clicks=0),
                html.Div(id='dropdown-container', children=[], style={'align': 'center', 'marginTop': '0.2in'}),
            ]
        ),
        dbc.Button("Confirm Preferences", id="confirm-preferences", n_clicks=0, className="d-grid gap-2", style={'marginTop': '0.2in'}),
        html.P(id='prefs-messy-succ', style={'color': 'green', 'marginTop': '0.2in'}),
        html.P(id='prefs-mess-warning', style={'color': 'red', 'marginTop': '0.2in'}),

    ], style={'marginTop': '0.2in', 'marginLeft': '0.5in', 'marginRight': '0.5in'}
    ),
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
    dbc.Tooltip(
            "NEED TO FILL THIS OUT HORIZ",
            target="horiz",
        ),
    dbc.Tooltip(
            "NEED TO FILL THIS OUT CARD",
            target="max-card",
        ),
], style={'textAlign': 'center'}
)

#