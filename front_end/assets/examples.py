import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback, dependencies
import pandas as pd
import numpy as np

df1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')

df1 = pd.concat([df1]*3, ignore_index=True)[['State']]

exgraph1 = dcc.Graph(
        figure=dict(
            data=[
                dict(
                    x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                    2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                    y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
                    350, 430, 474, 526, 488, 537, 500, 439],
                    name='Rest of world',
                    marker=dict(
                        color='rgb(55, 83, 109)'
                    )
                ),
                dict(
                    x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                    2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                    y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,
                    299, 340, 403, 549, 499],
                    name='China',
                    marker=dict(
                        color='rgb(26, 118, 255)'
                    )
                )
            ],
            layout=dict(
                title='US Export of Plastic Scrap',
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1.0
                ),
                margin=dict(l=40, r=0, t=40, b=30)
            )
        ),
        style={'height': 300},
        id='my-graph'
    )

exgraph2 = dcc.Graph(
        figure=dict(
            data=[
                dict(
                    x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                    2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                    y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
                    350, 430, 474, 526, 488, 537, 500, 439],
                    name='Rest of world',
                    marker=dict(
                        color='rgb(55, 83, 109)'
                    )
                ),
                dict(
                    x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                    2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                    y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,
                    299, 340, 403, 549, 499],
                    name='China',
                    marker=dict(
                        color='rgb(26, 118, 255)'
                    )
                )
            ],
            layout=dict(
                title='US Export of Plastic Scrap',
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1.0
                ),
                margin=dict(l=40, r=0, t=40, b=30)
            )
        ),
        style={'height': 700},
        id='my-graph'
    )

extable1 = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df1.columns],
    data=df1.to_dict('records'),
    css=[{'selector': '.row', 'rule': 'margin: 0'}],
    style_table={'overflowY': 'auto', 'overflowX': 'False', 'height': 550},
)

extable2 = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df2.columns],
    data=df2.to_dict('records')
)