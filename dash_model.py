# Note: using Github Copilot for the first time in a file.
# Also using https://towardsdatascience.com/build-a-machine-learning-simulation-tool-with-dash-b3f6fd512ad6 heavily as a guide.
# Thanks to the author Pierre-Louis Bescond. Check out the article for more information.

from distutils.log import debug
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

# Create dash app
app = dash.Dash()

# Page structure will be:
    # Distribution of values
    # Histogram of fitted model
    # Box to enter zip code
    # Checkbox for type of school
    # Slider to update ranking
    # Slider to update announce_date
    # Slider to update student body size
    # Map of results from fitted model on all counties with the given parameters

# apply basic HTML layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '100%', 'font-family': 'Verdana'}, 
children=[
    html.H1('Dashboard for Modeling'),
    # html.H3('Distribution of Values'),
    # dcc.Graph(id='distribution-of-values'),
    # show the number of boosters from callback
    html.H3('Number of Boosters'),
    html.H4(id='number-of-boosters'),    
    html.H3('Enter Zip Code'),
    dcc.Input(
        id='zip_code',
        placeholder='Enter Zip Code',        
        type='number',
        value=60091,
    ),
    html.H3('Type of School'),
    dcc.Checklist(
        id='type',
        options=[
            {'label': 'Public', 'value': 'Public'},
            {'label': 'Private', 'value': 'Private'},
        ],
        value=['Public'],        
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
    )    
])

# Create callback for histogram using all the values provided and the update_prediction method below
@app.callback(Output('number-of-boosters', 'children'), # Output('distribution-of-values', 'figure'), 
                [Input('zip_code', 'value'), Input('type', 'value'), Input('ranking', 'value'), Input('announce_date', 'value'), Input('student_body_size', 'value')])

# Note that Dash calls this method with default values when it starts.
def update_prediction(zip_code, type, ranking, announce_date, student_body_size):    
    """
    Updates histogram based on user-input. Note that in Sci-kit learn, the order of the columns matters. So, I need to transform my input.
    """
    # create dataframe with zip code, ranking, announce date, and student body size
    college_data = pd.DataFrame({'zip': [zip_code], 'ranking': [ranking], 'announce_date': [announce_date], 'Type': type}) # type is already a list      
    college_data['ranking'] = pd.cut(college_data['ranking'], bins=[0, 20, 100, 200, 298, 400], labels=['a', 'b', 'c', 'd', 'e'], right=False)  # cut the ranking into 5 bins
    college_data = cleaning(college_data, date_cols=None, last_tracking_date='3/25/2021', ignore_college=True)
    college_data.drop(columns=['zip', 'state', 'state_new', 'STCOUNTYFP', 'state_fips', 'county_fips', 'county_fips_str', 'State', 'State Code', 'Division'], 
               inplace=True)    
    college_data['2020.student.size'] = student_body_size # this is the last column for my sklearn features, so it also must be last here    
    college_data['booster'] = model.predict(college_data)    
    # create histogram for college_data['booster'] without using a separate method
    # histogram = go.Histogram(
    #     x=college_data['booster'],
    #     opacity=0.75,
    #     name='Booster',
    #     marker=dict(
    #         color='#FF0000',
    #         line=dict(
    #             color='#FF0000',
    #             width=1
    #         )
    #     )
    # )    
    return college_data['booster'].sum()

    # # create histogram with dash only for college_data
    # histogram = create_histogram(college_data)
    # # create figure with histogram and college_data
    # figure = go.Figure(data=[histogram])
    # return figure #college_data['booster'], hist_update #, map_update

if __name__ == '__main__':
    app.run_server(debug=True)    

# Note: need to get list of counties and their data so that I don't have to call the api everytime.