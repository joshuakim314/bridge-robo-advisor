import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies

from assets.nav import navbar
from assets.examples import exgraph1, exgraph2, extable1, extable2

main_dash = html.Div(

    [
        dbc.Row(
            [
                html.H2(id='test-callback', style={'textAlign':'center'}),
                dbc.Col(
                    [
                        dbc.Row(
                            html.Div(
                                [
                                    exgraph1
                                ],
                                         )),
                        dbc.Row(extable2)
                    ],
                    lg=9,
                    align='start'
                    ),
                dbc.Col(
                    [
                        extable1,
                        html.Div([dbc.Button("Deposit", color="primary", href='/nyi')], className="d-grid gap-2")
                    ],
                    lg=3,
                    align='end'
                    )
            ]
        ),
    ]
)

main_page = html.Div([
    navbar,
    html.Div([main_dash], style={'marginTop': '0.2in', 'marginLeft': '1.2in', 'marginRight': '1.2in'})
    ])

