import dash.exceptions
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

from app import app


email_input = html.Div(
    [
        dbc.Input(type="email", id="example-email", placeholder="Enter email"),
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

form = html.Div([
    dbc.Fade([
        dbc.Form([email_input, password_input]),
        dbc.Button("Continue", color="primary", href='/main'),
        ],
        id="fade-transition",
        is_in=False,
        style={"transition": "opacity 2000ms ease"},
        timeout=2000,
    )
], style={"width": "4in", "display": "inline-block", 'marginTop': '0.2in'}
)

#Appear Log In Info
@app.callback(
    Output(component_id='body-div', component_property='children'),
    Input(component_id='show-secret', component_property='n_clicks')
)
def update_output(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        return form

#Fade In Login Info
@app.callback(
    Output("fade-transition", "is_in"),
    [Input("show-secret", "n_clicks")],
    [State("fade-transition", "is_in")],
)
def toggle_fade(n, is_in):
    if not n:
        # Button has never been clicked
        return True
    return True

#Menu Bar Toggle
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

#Test Pass Pages
@app.callback(
    Output("memory-output", "data"),
    Input('continue', 'n_clicks'),
    [State("first-name", "value")]
)
def store_name(n_clicks, value):
        return str(value)

@app.callback(
    Output("memory-output", "data"),
    [Input("log-out", "n_clicks")],
    prevent_initial_call=True
)
def reset_name(n_clicks):
    if n_clicks > 0:
        print('Resetting Name')
        return None
    else:
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("test-callback", "children"),
    Input("memory-output", "data")
)
def get_welcome_string(value):
    return f'Welcome {value}, you currently have $12,345 invested'




