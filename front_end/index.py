import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform

from pages.welcome_page import welcome_page
from pages.sign_up_1 import sign_up_1
from pages.sign_up_2 import sign_up_2

from pages.main_page import main_page
from pages.deposit_page import deposit_page
from pages.prefs_page import prefs_page

from pages.not_yet_implemented import nyi_page

from pages.test_page import test_page
import callbacks
from app import app


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='account-info'),
    dcc.Store(id='portfolio-info'),
    dcc.Store(id='past-deposits'),
    dcc.Store(id='past-deposits-brief'),
    dcc.Store(id='portfolio-graph-data'),
    dcc.Store(id='portfolio-value'),
    dcc.Store(id='all-stock-list'),
    dcc.Store(id='trades'),
    dcc.Store(id='port-comp'),
    dcc.Store(id='portfolio-returns')
])

#Page Navigation
@app.callback(dependencies.Output('page-content', 'children'),
              [dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/sign_up':
        return sign_up_1
    if pathname == '/sign_up_cont':
        return sign_up_2
    if pathname == '/main':
        return main_page
    if pathname == '/deposit':
        return deposit_page
    if pathname == '/prefs':
        return prefs_page
    if pathname == '/nyi':
        return nyi_page
    if pathname == '/test':
        return test_page
    else:
        return welcome_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True)