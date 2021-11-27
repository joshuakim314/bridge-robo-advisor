import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies

from pages.welcome_page import welcome_page
from pages.sign_up import sign_up

from pages.test_page import test_page
import callbacks
from app import app

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

#Page Navigation
@app.callback(dependencies.Output('page-content', 'children'),
              [dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/sign_up':
        return sign_up
    if pathname == '/test':
        return test_page
    else:
        return welcome_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True)