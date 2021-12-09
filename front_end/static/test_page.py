import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies
from static.nav import navbar

test_page = html.Div([
    navbar,
    html.H1("Bootstrap Grid System Example")
        , dbc.Row(dbc.Col(html.Div(dbc.Alert("This is one column", color= "primary")))
        )
    , dbc.Row([
        dbc.Col(html.Div(dbc.Alert(
            [
                "Go Back To ",
                dcc.Link("Welcome Page", href="/", className="alert-link"),
            ],
            color="primary",
        ),))
        , dbc.Col(html.Div(dbc.Alert("One of three columns", color= "primary")))
        , dbc.Col(html.Div(dbc.Alert("One of three columns", color= "primary")))
        ])
    ])