# Note: using Github Copilot for the first time in a file.
# Also using https://towardsdatascience.com/build-a-machine-learning-simulation-tool-with-dash-b3f6fd512ad6 heavily as a guide.
# Thanks to the author Pierre-Louis Bescond. Check out the article for more information.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output
# import cleaning method from cleaning.py
from cleaning import cleaning


# import fitted model
# from sklearn.externals import joblib
import joblib
model = joblib.load('booster_log_model_jlib')
# model = joblib.load('booster_model.pkl')

# Create histogram with dash for fitted model
def create_histogram(df):
    histogram = go.Histogram(
        x=df['prediction'],
        opacity=0.75,
        name='Prediction',
        marker=dict(
            color='#FF0000',
            line=dict(
                color='#FF0000',
                width=1
            )
        )
    )
    return histogram

hist_importances = create_histogram(df)

# Create dash app
app = dash.Dash()

# Page structure will be:
    # Distribution of values
    # Histogram of fitted model
    # Box to enter zip code
    # Slider to update ranking
    # Slider to update announce_date
    # Slider to update student body size
    # Map of results from fitted model on all counties with the given parameters

# apply basic HTML layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '100%', 'font-family': 'Verdana'}, 
children=[
    html.H1('Dashboard for Modeling'),
    html.H3('Distribution of Values'),
    dcc.Graph(
        id='histogram',
        figure={
            'data': [hist_importances],
            'layout': {
                'title': 'Distribution of Values',
                'xaxis': {'title': 'Prediction'},
                'yaxis': {'title': 'Frequency'}
            }
        }
    ),
    html.H3('Enter Zip Code'),
    dcc.Input(
        id='zip_code',
        placeholder='Enter Zip Code',
        type='text',
        value=''
    ),
    html.H3('Update Ranking'),
    dcc.Slider(
        id='ranking',
        min=0,
        max=10,
        step=1,
        value=5,
        marks={i: str(i) for i in range(11)}
    ),
    html.H3('Update Announce Date'),
    dcc.Slider(
        id='announce_date',
        min=0,
        max=365,
        step=1,
        value=180,
        marks={i: str(i) for i in range(366)}
    ),
    html.H3('Update Student Body Size'),
    dcc.Slider(
        id='student_body_size',
        min=0,
        max=100,
        step=1,
        value=50,
        marks={i: str(i) for i in range(101)}
    ),
    html.H3('Map of Results'),
    # dcc.Graph(
    #     id='map',
    #     figure={
    #         'data': [go.Scattermapbox(
    #             lat=df['lat'],
    #             lon=df['lon'],
    #             mode='markers',
    #             marker=dict(
    #                 size=df['prediction'] * 10,
    #                 color=df['prediction'],
    #                 colorscale='Viridis',
    #                 showscale=True
    #             ),
    #             text=df['prediction'],                                
    #             hoverinfo='text'
    #         )],
    #         'layout': go.Layout(
    #             title='Map of Results',
    #             autosize=True,
    #             hovermode='closest',
    #             mapbox=dict(
    #                 accesstoken=mapbox_access_token,
    #                 bearing=0,
    #                 center=dict(
    #                     lat=37.7749,
    #                     lon=-122.4194
    #                 ),
    #                 pitch=0,
    #                 zoom=10
    #             ),
    #             showlegend=False
    #         )
    #     }
    # )
])

@app.callback(Output('map', 'figure'), [Input('zip_code', 'value'), Input('ranking', 'value'), 
                                        Input('announce_date', 'value'), Input('student_body_size', 'value')])

def update_prediction(zip_code, ranking, announce_date, student_body_size):
    # create dataframe with zip code, ranking, announce date, and student body size
    df_update = pd.DataFrame({'zip': zip_code, 'ranking': ranking, 'announce_date': announce_date, 'size': student_body_size})
    # call cleaning from cleaning.py
    df_update = cleaning(df_update, last_tracking_date='3/25/2021', ignore_college=True)
    # predict using fitted model
    df_update['prediction'] = model.predict(df_update)
    # create histogram with dash for fitted model
    hist_update = create_histogram(df_update)
    # create map with dash for fitted model
    # map_update = go.Scattermapbox(
    #     lat=df_update['lat'],
    #     lon=df_update['lon'],
    #     mode='markers',
    #     marker=dict(
    #         size=df_update['prediction'] * 10,
    #         color=df_update['prediction'],
    #         colorscale='Viridis',
    #         showscale=True
    #     ),
    #     text=df_update['prediction'],                                
    #     hoverinfo='text'
    # )
    return hist_update #, map_update

if __name__ == '__main__':
    app.run_server(debug=True)    