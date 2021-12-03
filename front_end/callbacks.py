import collections
import plotly.express as px
import plotly.graph_objects as go

import dash.exceptions
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies, no_update, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import time
import pandas as pd

from backend_access import push_new_user, pull_user_data, update_risk_control, add_transaction, get_portfolio_data, get_past_deposits, get_past_deposits_brief, get_portfolio_value_by_date

from app import app

import numpy as np
import psycopg2.extensions
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

conn = psycopg2.connect(
    host='database-1.csuf8nkuxrw3.us-east-2.rds.amazonaws.com',
    port=5432,
    user='postgres',
    password='capstone',
    database='can2_etfs'
)

email_input = html.Div([
        dbc.Input(type="email", id="login-email", placeholder="Enter email"),
    ],className="mb-3")

password_input = html.Div([
        dbc.Input(
            type="password",
            id="login-password",
            placeholder="Enter password",
        ),
    ],className="mb-3",)



form = html.Div([
    dbc.Fade([
        dbc.Form([email_input, password_input]),
        dbc.Button("Continue", id='login', color="primary"),
        html.P(id='invalid', style={'color': 'red', 'marginTop': '0.2in'})
        ],
        id="fade-transition",
        is_in=False,
        style={"transition": "opacity 2000ms ease"},
        timeout=2000,
    )
], style={"width": "4in", "display": "inline-block", 'marginTop': '0.2in'})

depositForm = html.Div([
    dbc.Fade([
        dbc.Form([
            dbc.InputGroup(
                [
                    dbc.InputGroupText("$"),
                    dbc.Input(placeholder="Deposit Amount", type="number", id='deposit-ammount')
                ],
                className="mb-3",
            ),
            ]
        ),
        dbc.Button("Confirm", color="primary", id='confirm-deposit', n_clicks=0),
        ],
        id="deposit-fade-transition",
        is_in=False,
        style={"transition": "opacity 2000ms ease"},
        timeout=2000,
    )
], style={"width": "4in", "display": "inline-block", 'marginTop': '0.2in'})

withdrawForm = html.Div([
    dbc.Fade([
        dbc.Form([
            dbc.InputGroup(
                [
                    dbc.InputGroupText("-$"),
                    dbc.Input(placeholder="Withdraw Amount", type="number", id='withdraw-ammount')
                ],
                className="mb-3",
            ),
            ]
        ),
        dbc.Button("Confirm", color="danger", id='confirm-withdraw'),
        ],
        id="withdraw-fade-transition",
        is_in=False,
        style={"transition": "opacity 2000ms ease"},
        timeout=2000,
    )
], style={"width": "4in", "display": "inline-block", 'marginTop': '0.2in'})

#Create User Profile
@app.callback(
    [Output(component_id='account-info', component_property='data'),
     Output(component_id='invalid', component_property='children'),
     Output(component_id='url', component_property='pathname')],
    Input(component_id='continue', component_property='n_clicks'),
    [State(component_id='first-name-su', component_property='value'),
     State(component_id='last-name-su', component_property='value'),
     State(component_id='email-su', component_property='value'),
     State(component_id='check-email-su', component_property='value'),
     State(component_id='password-su', component_property='value'),
     State(component_id='check-password-su', component_property='value')]
)
def new_user_data(n_clicks, fn, ln, em, cem, pw, cpw):
    try:
        if len(fn) > 0 and len(ln) > 0 and len(em) and len(cem) and len(pw) and len(cpw):
            try:
                if (em == cem) and (pw == cpw):
                    push_new_user(conn, np.array([fn, ln, em, pw, 0, 0]))
                    user_data = pull_user_data(conn, em, pw)
                    add_transaction(conn, em, 'cash', 0, 'Initialize Account')
                    #print(user_data)
                    return user_data, no_update, '/sign_up_cont'
                else:
                    return no_update, 'Email or password do not match', no_update

            except:
                #print('Failed Retreiving Password')
                return no_update, 'Error Occured', no_update
    except:
        raise dash.exceptions.PreventUpdate

#Add Risk User Profile
@app.callback(
    [Output(component_id='account-info', component_property='data'),
     Output(component_id='invalid', component_property='children'),
     Output(component_id='url', component_property='pathname')],
    Input(component_id='continue-2', component_property='n_clicks'),
    [State(component_id='user-risk', component_property='value'),
     State(component_id='user-control', component_property='value'),
     State(component_id='account-info', component_property='data')]
)
def update_user_risk_data(n_clicks, ur, uc, acc_info):
    #try:
    if n_clicks != 0:
        #print(acc_info['Email'], acc_info['Password'], ur, uc)
        update_risk_control(conn, acc_info['Email'], acc_info['Password'], ur, uc)
        user_data = pull_user_data(conn, acc_info['Email'], acc_info['Password'])
        #print(user_data)
        return user_data, no_update, '/main'
    else:
        return no_update, no_update, no_update
    #except:
        #raise dash.exceptions.PreventUpdate


#Login and Grab User Info
@app.callback(
    [Output(component_id='account-info', component_property='data'),
     Output(component_id='portfolio-info', component_property='data'),
     Output(component_id='past-deposits', component_property='data'),
     Output(component_id='past-deposits-brief', component_property='data'),

     Output(component_id='portfolio-graph-data', component_property='data'),
     Output(component_id='portfolio-value', component_property='data'),

     Output(component_id='invalid', component_property='children'),
     Output(component_id='url', component_property='pathname')],
    Input(component_id='login', component_property='n_clicks'),
    [State(component_id='login-email', component_property='value'),
     State(component_id='login-password', component_property='value')],
    prevent_initial_call=True
)
def get_user_data(n_clicks, email, password):
    #print(email)
    #print(password)
    try:
        if len(email) > 0 and len(password) > 0:
            try:
                user_data = pull_user_data(conn, email, password)
                portfolio_data = get_portfolio_data(conn, email)
                past_deposits = get_past_deposits(conn, email)
                past_deposits_brief = get_past_deposits_brief(conn, email)
                portfolio_graph_data, portfolio_value = get_portfolio_value_by_date(conn, email)
                #print('Got Past Deposits')
                return user_data, portfolio_data, past_deposits, past_deposits_brief, portfolio_graph_data, portfolio_value, no_update, '/main'
            except:
                #print('Failed Retreiving Password')
                return no_update, no_update, no_update, no_update, no_update, no_update, 'Username or Password does not match an account on file', no_update
    except:
        raise dash.exceptions.PreventUpdate

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

#Appear Deposit Form
@app.callback(
    Output(component_id='deposit-secret-div', component_property='children'),
    Input(component_id='deposit-show-secret', component_property='n_clicks')
)
def update_output_deposit(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        return depositForm

#Fade In Login Info
@app.callback(
    Output("deposit-fade-transition", "is_in"),
    [Input("deposit-show-secret", "n_clicks")],
    [State("deposit-fade-transition", "is_in")],
)
def toggle_fade_deposit(n, is_in):
    if not n:
        # Button has never been clicked
        return True
    return True

#Appear Withdraw Form
@app.callback(
    Output(component_id='withdraw-secret-div', component_property='children'),
    Input(component_id='withdraw-show-secret', component_property='n_clicks')
)
def update_output_withdraw(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        return withdrawForm

#Fade In Login Info
@app.callback(
    Output("withdraw-fade-transition", "is_in"),
    [Input("withdraw-show-secret", "n_clicks")],
    [State("withdraw-fade-transition", "is_in")],
)
def toggle_fade_withdraw(n, is_in):
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

#Display User Info Header
@app.callback(
    Output("total-deposit-info", "children"),
    [Input("account-info", "data"),
    Input("portfolio-info", "data")]
)
def toggle_navbar_collapse(account_data, portfolio_data):
    try:
        ret_string = f'Hi {account_data["First"]}, the cash component of your portfolio equals ${portfolio_data[0]["Amount"]/100:,.2f}'
        return ret_string
    except:
        raise PreventUpdate

#Dynamically add stock prefs
@app.callback(
    Output('dropdown-container', 'children'),
    Input('add-filter', 'n_clicks'),
    State('dropdown-container', 'children'))
def display_dropdowns(n_clicks, children):
    new_dropdown = html.Div([
        dcc.Dropdown(
        id={
            'type': 'filter-dropdown',
            'index': n_clicks
        },
        options=[{'label': i, 'value': i} for i in ['NYC', 'MTL', 'LA', 'TOKYO']]
        ),
        dcc.RangeSlider(id="input-range", min=0, max=100, value=[0, 100])
    ], style={"width": "4in", "display": "inline-block", 'marginTop': '0.05in'}
    )
    children.append(new_dropdown)

    return children

@app.callback(
    Output('dropdown-container-output', 'children'),
    [Input({'type': 'filter-dropdown', 'index': ALL}, 'value'),
     Input({'type': 'input-range', 'index': ALL}, 'value')]
)
def display_output(values, ranges):
    stocks = html.Div([
        html.Div('Dropdown {} = {}'.format(i + 1, value))
        for (i, value) in enumerate(values)
    ])
    ranges = html.Div([
        html.Div('Dropdown {} = {}'.format(i + 1, ranges))
        for (i, ranges) in enumerate(ranges)
    ])
    return html.Div([stocks, ranges])

#Deposit Handle
@app.callback(
    [Output('portfolio-info', 'data'),
     Output('past-deposits', 'data'),
     Output('past-deposits-brief', 'data'),
     Output(component_id='portfolio-graph-data', component_property='data'),
     Output(component_id='portfolio-value', component_property='data')],
    Input('confirm-deposit', 'n_clicks'),
    [State('deposit-ammount', 'value'),
     State('account-info', 'data')]
)
def handle_deposit(n_clicks, deposit_amount, account_info):
    try:
        if deposit_amount > 0:
            add_transaction(conn, account_info['Email'], 'cash', deposit_amount*100, f'Deposited ${deposit_amount:,.2f}')
            portfolio_data = get_portfolio_data(conn, account_info['Email'])
            deposit_data = get_past_deposits(conn, account_info['Email'])
            deposit_data_brief = get_past_deposits_brief(conn, account_info['Email'])
            portfolio_graph_data, portfolio_value = get_portfolio_value_by_date(conn, account_info['Email'])
            return portfolio_data, deposit_data, deposit_data_brief, portfolio_graph_data, portfolio_value
        deposit_data = get_past_deposits(conn, account_info['Email'])
        deposit_data_brief = get_past_deposits_brief(conn, account_info['Email'])
        return no_update, deposit_data, deposit_data_brief, no_update, no_update
    except:
        raise PreventUpdate

#Withdraw Handle
@app.callback(
    [Output('portfolio-info', 'data'),
     Output('past-deposits', 'data'),
     Output('past-deposits-brief', 'data'),
     Output(component_id='portfolio-graph-data', component_property='data'),
     Output(component_id='portfolio-value', component_property='data')],
    Input('confirm-withdraw', 'n_clicks'),
    [State('withdraw-ammount', 'value'),
     State('account-info', 'data')]
)
def handle_withdraw(n_clicks, deposit_amount, account_info):
    try:
        if deposit_amount > 0:
            add_transaction(conn, account_info['Email'], 'cash', -1*deposit_amount*100, f'Withdrew ${deposit_amount:,.2f}')
            portfolio_data = get_portfolio_data(conn, account_info['Email'])
            deposit_data = get_past_deposits(conn, account_info['Email'])
            deposit_data_brief = get_past_deposits_brief(conn, account_info['Email'])
            portfolio_graph_data, portfolio_value = get_portfolio_value_by_date(conn, account_info['Email'])
            return portfolio_data, deposit_data, deposit_data_brief, portfolio_graph_data, portfolio_value
        deposit_data = get_past_deposits(conn, account_info['Email'])
        deposit_data_brief = get_past_deposits_brief(conn, account_info['Email'])
        return no_update, deposit_data, deposit_data_brief, no_update, no_update
    except:
        raise PreventUpdate

#Get deposits table
@app.callback(
    Output('deposits-table', 'data'),
    Input('past-deposits', 'data')
)

def print_deposits_table(past_deposits):
    try:
        return past_deposits
    except:
        raise PreventUpdate

#Get shorter deposits table
@app.callback(
    Output('deposits-table-brief', 'data'),
    Input('past-deposits-brief', 'data')
)

def print_deposits_table_brief(past_deposits):
    try:
        return past_deposits
    except:
        raise PreventUpdate

#Show Main Page Graph and Header
@app.callback(
    [Output('portfolio-graph', 'figure'),
     Output('portfolio-highlight', 'children')],
    [Input('account-info', 'data'),
     Input('portfolio-value', 'data'),
     Input('portfolio-graph-data', 'data')]
)
def get_main_page_portfolio_info(account_data, account_value, data):

    dff = pd.read_json(data, orient='split')
    #print(dff)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff['date'], y=dff['value'], line=dict(color="#66A593")))
    #figure = px.line(dff, x='date', y='value', title='Portfolio Value', template='plotly_white', color='green')
    ret_string = f'Hi {account_data["First"]}, you currently have ${account_value["portfolio_value"]:,.2f} invested'

    fig.update_layout(
        title="Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white"
    )
    fig.update_yaxes(tickprefix="$")

    return fig, ret_string

