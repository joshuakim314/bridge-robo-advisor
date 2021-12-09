import dash_bootstrap_components as dbc
from dash import Dash, State, html, dcc, dash_table, callback, dependencies
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform

from static.nav import navbar

deposit_page = html.Div([
    navbar,
    html.H2(id='test-callback', style={'marginTop': '0.2in', 'marginLeft': '1.2in', 'marginRight': '1.2in', 'textAlign': 'center'}),
    html.Div([
            dbc.Button("Deposit", id='deposit-show-secret', color="primary", outline=False, className="me-3", size='lg'),
            dbc.Button("Withdraw", id='withdraw-show-secret', color="danger", outline=True, className="me-3", size='lg'),
            html.Div(id='deposit-secret-div'),
            html.Div(id='withdraw-secret-div'),
            html.H2(id='total-deposit-info', style={'textAlign':'center', 'marginTop': '0.2in'}),
            html.H2(['Cash Transactions History'], style={'marginTop': '0.2in', 'textAlign': 'left'}),
            dash_table.DataTable(
                id='deposits-table',
                columns=[{'name': 'Date Time', 'id': 'Date Time'}, {'name': 'Transaction', 'id': 'Comment'}],
                css=[{'selector': '.row', 'rule': 'margin: 0'}],
            )
        ], style={'marginTop': '0.2in', 'marginLeft': '1.2in', 'marginRight': '1.2in', 'textAlign': 'center'}
    ),


],
    #className="d-grid gap-0 col-2 mx-auto",
)

