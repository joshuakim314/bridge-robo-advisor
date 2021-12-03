import dash_bootstrap_components as dbc
from dash import Dash, State, html, dcc, dash_table, callback, dependencies
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform

from assets.nav import navbar
from assets.examples import exgraph1, exgraph2, extable1, extable2

main_dash = html.Div(

    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row([
                            html.H3(id='portfolio-highlight', style={'textAlign': 'center'}),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id='portfolio-graph',
                                        figure={}
                                    )
                                ],
                            )
                        ]
                        ),
                    ],
                    lg=9,
                    align='start'
                    ),
                dbc.Col(
                    [
                        html.H3('Past Deposits'),
                        dash_table.DataTable(
                            id='deposits-table-brief',
                            columns=[{'name': 'Date', 'id': 'Date Time'}, {'name': 'Amount', 'id': 'Amount'}],
                            css=[{'selector': '.row', 'rule': 'margin: 0'}],
                            style_table={'overflowY': 'auto', 'overflowX': 'False', 'height': 500},
                        ),
                        html.Div([dbc.Button("Deposit", color="primary", href='/deposit')], className="d-grid gap-2")
                    ],
                    lg=3,
                    align='top'
                    )
            ]
        ),
    ]
)

main_page = html.Div([
    navbar,
    html.Div([main_dash], style={'marginTop': '0.2in', 'marginLeft': '1.2in', 'marginRight': '1.2in'})
    ])

'''
dash_table.DataTable(
                            id='deposits-table-brief',
                            columns=[{'name': 'Date Time', 'id': 'Date Time'}, {'name': 'Amount', 'id': 'Amount'}],
                            css=[{'selector': '.row', 'rule': 'margin: 0'}],
                            style_table={'overflowY': 'True', 'height': 550},
                        ),
'''