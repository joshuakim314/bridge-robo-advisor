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

from backend_access import push_new_user, pull_user_data, update_risk_control, add_transaction, get_portfolio_data, get_past_deposits, get_past_deposits_brief, get_portfolio_value_by_date, get_all_tickers, get_portfolio_weights, get_trades, transact_stock

from app_old import app

import numpy as np
import psycopg2.extensions
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

import static.portfolio_opt_front_end as portfolio_opt_front_end

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
     Output(component_id='url', component_property='pathname'),
     Output(component_id='all-stock-list', component_property='data')],
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
        if len(fn) > 0 and len(ln) > 0 and len(em) > 0 and len(cem) > 0 and len(pw) > 0 and len(cpw) > 0:
            try:
                if (em == cem) and (pw == cpw):
                    push_new_user(conn, np.array([fn, ln, em, pw, 70, 50, 20, 2, 50]))
                    user_data = pull_user_data(conn, em, pw)
                    add_transaction(conn, em, 'cash', 0, 'Initialize Account')
                    stock_list = get_all_tickers(conn)
                    #print(stock_list)
                    #print(user_data)
                    return user_data, no_update, '/sign_up_cont', stock_list
                else:
                    return no_update, 'Email or password do not match', no_update, no_update

            except:
                #print('Failed Retreiving Password')
                return no_update, 'Error Occured', no_update, no_update
    except:
        raise dash.exceptions.PreventUpdate

#Add Risk User Profile
@app.callback(
    [Output(component_id='account-info', component_property='data'),
     Output(component_id='invalid', component_property='children'),
     Output(component_id='url', component_property='pathname'),
     Output(component_id='portfolio-graph-data', component_property='data'),
     Output(component_id='portfolio-value', component_property='data'),
     Output(component_id='portfolio-returns', component_property='data'),
     Output(component_id='port-comp', component_property='data')],
    Input(component_id='continue-2', component_property='n_clicks'),
    [State(component_id='user-risk', component_property='value'),
     State(component_id='user-control', component_property='value'),
     State(component_id='user-horizon', component_property='value'),
     State(component_id='user-return', component_property='value'),
     State(component_id='account-info', component_property='data')]
)
def update_user_risk_data(n_clicks, ur, uc, uh, uret, acc_info):
    try:
        if n_clicks != 0:
            #print(acc_info['Email'], acc_info['Password'], ur, uc)
            update_risk_control(conn, acc_info['Email'], acc_info['Password'], ur, uc, uh, uret, 35)
            user_data = pull_user_data(conn, acc_info['Email'], acc_info['Password'])
            portfolio_graph_data, portfolio_value, port_returns, port_comp = get_portfolio_value_by_date(conn, acc_info['Email'])
            #print(user_data)
            return user_data, no_update, '/main', portfolio_graph_data, portfolio_value, port_returns, port_comp
        else:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    except:
        raise dash.exceptions.PreventUpdate


#Login and Grab User Info
@app.callback(
    [Output(component_id='account-info', component_property='data'),
     Output(component_id='portfolio-info', component_property='data'),
     Output(component_id='past-deposits', component_property='data'),
     Output(component_id='past-deposits-brief', component_property='data'),

     Output(component_id='portfolio-graph-data', component_property='data'),
     Output(component_id='portfolio-value', component_property='data'),
     Output(component_id='portfolio-returns', component_property='data'),
     Output(component_id='all-stock-list', component_property='data'),

     Output(component_id='invalid', component_property='children'),
     Output(component_id='url', component_property='pathname'),
     Output(component_id='port-comp', component_property='data')],

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
                portfolio_graph_data, portfolio_value, port_returns, port_comp = get_portfolio_value_by_date(conn, email)
                all_stocks = get_all_tickers(conn)
                #print('Got Past Deposits')
                return user_data, portfolio_data, past_deposits, past_deposits_brief, portfolio_graph_data, portfolio_value, port_returns, all_stocks, no_update, '/main', port_comp
            except:
                #print('Failed Retreiving Password')
                return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, 'Username or Password does not match an account on file', no_update, no_update
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

#Deposit Handle
@app.callback(
    [Output('portfolio-info', 'data'),
     Output('past-deposits', 'data'),
     Output('past-deposits-brief', 'data'),
     Output(component_id='portfolio-graph-data', component_property='data'),
     Output(component_id='portfolio-value', component_property='data'),
     Output(component_id='portfolio-returns', component_property='data'),

     Output(component_id='port-comp', component_property='data')],
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
            portfolio_graph_data, portfolio_value, port_returns, port_comp = get_portfolio_value_by_date(conn, account_info['Email'])
            return portfolio_data, deposit_data, deposit_data_brief, portfolio_graph_data, portfolio_value, port_returns, port_comp
        deposit_data = get_past_deposits(conn, account_info['Email'])
        deposit_data_brief = get_past_deposits_brief(conn, account_info['Email'])
        return no_update, deposit_data, deposit_data_brief, no_update, no_update, no_update, no_update
    except:
        raise PreventUpdate

#Withdraw Handle
@app.callback(
    [Output('portfolio-info', 'data'),
     Output('past-deposits', 'data'),
     Output('past-deposits-brief', 'data'),
     Output(component_id='portfolio-graph-data', component_property='data'),
     Output(component_id='portfolio-value', component_property='data'),
     Output(component_id='portfolio-returns', component_property='data'),

     Output(component_id='port-comp', component_property='data')],
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
            portfolio_graph_data, portfolio_value, port_returns, port_comp = get_portfolio_value_by_date(conn, account_info['Email'])
            return portfolio_data, deposit_data, deposit_data_brief, portfolio_graph_data, portfolio_value, port_returns, port_comp
        deposit_data = get_past_deposits(conn, account_info['Email'])
        deposit_data_brief = get_past_deposits_brief(conn, account_info['Email'])
        return no_update, deposit_data, deposit_data_brief, no_update, no_update, no_update, no_update
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

#Set values of account parameters in prefs page
@app.callback(
    [Output('user-risk-set', 'value'),
     Output('user-return-set', 'value'),
     Output('user-control-set', 'value'),
     Output('user-horizon-set', 'value'),
     Output('user-card-set', 'value'),
     Output('account-info', 'data')],
    Input('confirm-preferences', 'n_clicks'),
    [State('user-risk-set', 'value'),
     State('user-return-set', 'value'),
     State('user-control-set', 'value'),
     State('user-horizon-set', 'value'),
     State('user-card-set', 'value'),
     State('account-info', 'data')]
)
def set_account_params_vals(n_clicks, urisk, ureturn, ucontrol, uhorizon, max_card, data):
    try:
        update_risk_control(conn, data['Email'], data['Password'], urisk, ucontrol, uhorizon, ureturn, max_card)

        data['Risk'] = urisk
        data['Return'] = ureturn
        data['Control'] = ucontrol
        data['Horizon'] = uhorizon
        data['Max'] = max_card

        return no_update, no_update, no_update, no_update, no_update, data
    except:
        try:
            return data['Risk'], data['Return'], data['Control'], data['Horizon'], data['Max'], no_update
        except:
            raise PreventUpdate

#Dynamically add stock picks
@app.callback(
    Output('dropdown-container', 'children'),
    Input('add-filter', 'n_clicks'),
    [State('dropdown-container', 'children'),
     State('all-stock-list', 'data')])
def display_dropdowns(n_clicks, children, stock_list):
    new_dropdown = html.Div([
        dcc.Dropdown(
            id={
                'type': 'stock-pick',
                'index': n_clicks
            },
            options=[{'label': i, 'value': i} for i in stock_list['stocks']]
            ),
        dcc.RangeSlider(
            id={
                'type': 'stock-min-max',
                'index': n_clicks
            },
            min=0, max=100, value=[0, 100],
            marks={
                0: {'label': '0%', 'style': {'color': '#313131'}},
                100: {'label': '100%', 'style': {'color': '#313131'}}
            },
            tooltip={"placement": "bottom", "always_visible": False}
            )
    ], style={"width": "3in", "display": "inline-block", 'marginTop': '0.05in'}
    )
    children.append(new_dropdown)

    return children

#Confirm Stock Pick
@app.callback(
    [Output('prefs-messy-succ', 'children'),
     Output('prefs-mess-warning', 'children'),
     Output('show-hype-graph', 'children'),
     Output('trades', 'data')],
    Input('confirm-preferences', 'n_clicks'),
    [State({'type': 'stock-pick', 'index': ALL}, 'value'),
     State({'type': 'stock-min-max', 'index': ALL}, 'value'),
     State('user-risk-set', 'value'),
     State('user-return-set', 'value'),
     State('user-control-set', 'value'),
     State('user-horizon-set', 'value'),
     State('user-card-set', 'value'),
     State('account-info', 'data')]
)
def display_output(n_clicks, values, ranges, urisk, uret, ucontr, uhoriz, ucard, account_data):
    #try:
        stocks = {}
        for (i, stk) in enumerate(values):
            #print(f'{i}: {stk}')
            stocks[i] = stk
        final = {}
        for (i, ran) in enumerate(ranges):
            #print(f'{i}: {ran}')
            final[i] = {'stock': stocks[i],
                        'range': ran}

        #Checks for invalid constraints:
        min = 0
        max = 0
        seen_stocks = set()
        for key in final.keys():
            #print(final[key]['stock'])
            if final[key]['stock'] in seen_stocks:
                return None, f'Duplicate stocks {final[key]["stock"]}, please change one or reload page to reset', no_update, no_update
            seen_stocks.add(final[key]["stock"])
            min += final[key]['range'][0]
            max += final[key]['range'][1]

        if len(seen_stocks) == 0 or all(x is None for x in list(seen_stocks)):
            return None, 'Please select at least one stock.', no_update, no_update

        if min > 100:
            return None, 'Minimum values resulting in infeasible constraints, please lower them.', no_update, no_update

        if max < ucontr:
            return None, 'Maximum values resulting in infeasible constraints, please raise them or lower User Portfolio Control.', no_update, no_update

        print(final)
        print(account_data)

        stock_picks = []

        weight_constraints = {}
        for key in final.keys():
            stock_picks.append(final[key]['stock'])
            weight_constraints[final[key]['stock']] = (final[key]['range'][0], final[key]['range'][1])

        control = account_data['Control']/100
        trade_horizon = account_data['Horizon']
        cardinality = account_data['Max']
        target_return = account_data['Return']/100
        risk_aversion = 10 - account_data['Risk']/10
        print(weight_constraints)


        optimal_portfolio, expected_monthly_returns = portfolio_opt_front_end.port_opt(stock_picks, weight_constraints, control, trade_horizon, cardinality, target_return, risk_aversion)

        print(optimal_portfolio)
        print(expected_monthly_returns)

        bt, fu, fm, fd, trades, old_portfolio, new_portfolio = get_trades(conn, account_data['Email'], optimal_portfolio, expected_monthly_returns)

        print(optimal_portfolio)
        print(expected_monthly_returns)

        #Get Historic and Expected Annualized Returns
        btmin = bt['value'].to_list()[0]
        btmax = bt['value'].to_list()[-1]

        bt_annual_returns = (1+((btmax-btmin)/btmin))**(0.2) - 1

        fmmin = fm['value'].to_list()[0]
        fmmax = fm['value'].to_list()[-1]

        fm_annual_returns = ((fmmax-fmmin)/fmmin)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt['date'], y=bt['value'], line=dict(color="#205445"), name='Historic Value'))
        fig.add_trace(go.Scatter(x=fu['date'], y=fu['value'], line=dict(color="#9fcfc1"), name='Expected Value Upper Bound'))
        fig.add_trace(go.Scatter(x=fm['date'], y=fm['value'], line=dict(color="#66A593"), name='Expected Value'))
        fig.add_trace(go.Scatter(x=fd['date'], y=fd['value'], line=dict(color="#9fcfc1"), name='Expected Value Lower Bound'))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Theoretical Portfolio Value",
            template="plotly_white",
            margin={'t': 20}
        )
        fig.update_yaxes(tickprefix="$")

        print(optimal_portfolio)

        pie_old = px.pie(values=old_portfolio['value'], names=old_portfolio['ticker'], color_discrete_sequence=px.colors.sequential.Greens)
        pie_old.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
        pie_old.update(layout_showlegend=False)

        pie_new = px.pie(values=new_portfolio['value'], names=new_portfolio['ticker'], color_discrete_sequence=px.colors.sequential.Greens)
        pie_new.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
        pie_new.update(layout_showlegend=False)

        retDiv = html.Div([
            html.H3(f'Theoretical portfolio has historic annualized returns of {bt_annual_returns:.2%} and expected annualized returns of {fm_annual_returns:.2%}', style={'marginLeft': '1in', 'marginRight': '1in', 'textAlign': 'left'}),
            dcc.Graph(
                figure=fig,
            ),
            dbc.Row([
                dbc.Col([
                    html.H5('Current Portfolio Composition'),
                    dcc.Graph(
                        figure=pie_old,
                    ),
                ]),
                html.Img(src='/static/arrow.png', style={'height': '0.5in', 'width': '1in', 'align': 'center', 'marginTop': '2in'}),
                dbc.Col([
                    html.H5('New Portfolio Composition'),
                    dcc.Graph(
                        figure=pie_new,
                    ),
                ])
            ]),
            dbc.Button("Make Trades for New Portfolio", id="confirm-trades", n_clicks=0),
            html.P(id='trade-messy-succ', style={'color': 'green', 'marginTop': '0.2in'}),
            html.P(id='trade-mess-warning', style={'color': 'red', 'marginTop': '0.2in'}),

        ])

        return 'Stock Pick Success', None, retDiv, trades

    #except:
        #raise PreventUpdate

@app.callback(
    [Output('trade-messy-succ', 'children'),
     Output('trade-mess-warning', 'children'),
     Output('trades', 'data'),
     Output(component_id='portfolio-info', component_property='data'),
     Output(component_id='past-deposits', component_property='data'),
     Output(component_id='past-deposits-brief', component_property='data'),
     Output(component_id='portfolio-graph-data', component_property='data'),
     Output(component_id='portfolio-value', component_property='data'),
     Output(component_id='portfolio-returns', component_property='data'),

     Output(component_id='port-comp', component_property='data')],

    Input('confirm-trades', 'n_clicks'),
    [State('trades', 'data'),
     State('account-info', 'data')]

)
def submit_trades(n_clicks, trades, ac_info):
    try:
        if n_clicks > 0:
            print(trades)
            if trades != None:
                for trade in trades:
                    #time.sleep(1)
                    if trade['delta'] != 0:
                        transact_stock(conn, ac_info['Email'], trade['ticker'], trade['delta'])
                portfolio_data = get_portfolio_data(conn, ac_info['Email'])
                past_deposits = get_past_deposits(conn, ac_info['Email'])
                past_deposits_brief = get_past_deposits_brief(conn, ac_info['Email'])
                portfolio_graph_data, portfolio_value, port_returns, port_comp = get_portfolio_value_by_date(conn, ac_info['Email'])

                return 'Trades Submitted', no_update, None, portfolio_data, past_deposits, past_deposits_brief, portfolio_graph_data, portfolio_value, port_returns, port_comp
            else:
                return None, 'Trades Already Submitted', no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        else:
            raise PreventUpdate
    except:
        raise PreventUpdate

@app.callback(
    Output('port-comp-pie', 'children'),
    Input('portfolio-highlight', 'children'),
    State('port-comp', 'data')
)
def display_port_comp_pie(chil, data):
    #try:
        pie_old = px.pie(values=list(data.values()), names=list(data.keys()), color_discrete_sequence=px.colors.sequential.Greens)
        pie_old.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=2)))
        pie_old.update(layout_showlegend=False)


        pie_old.update_layout(
            margin={'t': 4, 'l': 4, 'r': 4, 'b': 4}
        )

        tempDiv = dcc.Graph(
            figure=pie_old,
        )
        return tempDiv
    #except:
        raise PreventUpdate

#Get Portfolio Returns to Front Page
@app.callback(
    Output('port-ret-message', 'children'),
    Input('portfolio-highlight', 'children'),
    State('portfolio-returns', 'data')
)

def ret_port_rets(chil, data):
    if data['port_returns'] != 0:
        return f"Your portfolio's anualized returns are {data['port_returns']:.1%}"
    else:
        return None