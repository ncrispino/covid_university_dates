# Note: using Github Copilot for the first time in a file.
# Also using https://towardsdatascience.com/build-a-machine-learning-simulation-tool-with-dash-b3f6fd512ad6 heavily as a guide.
# Thanks to the author Pierre-Louis Bescond. Check out the article for more information.

from distutils.log import debug
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import plotly.express as px

# import fitted model
# from sklearn.externals import joblib
import joblib
model = joblib.load('booster_dummy_model_jlib')
# model = joblib.load('booster_log_model_jlib')
# model = joblib.load('booster_model.pkl')

# Create dash app
app = dash.Dash()

# Page structure will be:
    # Map of US by county shaded based on output from fitted model
    # Bar graph of output from fitted model
    # Total number of booster mandates
    # Slider for type of school, with two options
    # Slider to update ranking with 5 bins of range bins=[0, 20, 100, 200, 298, 400]
    # Slider to update announce_date
    # Slider to update student body size

# apply basic HTML layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '100%', 'font-family': 'Verdana'}, 
children=[
    html.H1('Dashboard for Modeling'),
    # show map from callback
    dcc.Graph(id='map'), #, style={'width': '100%', 'height': '100%'}),
    html.H3('Distribution of Values'),
    dcc.Graph(id='distribution-of-values'),
    # show the number of boosters from callback
    html.H3('Number of Boosters'),
    html.H4(id='number-of-boosters'),    
    html.H3('Type of School'),
    dcc.RadioItems(
        id='type',
        options=[
            {'label': 'Public', 'value': 'Public'},
            {'label': 'Private', 'value': 'Private'},
        ],
        value='Public',
        labelStyle={'display': 'inline-block'},
    ),            
    html.H3('Update Ranking'),
    dcc.Slider(
        id='ranking',
        min=0,
        max=400,
        step=1,
        value=100,
        marks={
            0: '0',
            20: '20',
            100: '100',
            200: '200',
            298: '298',
            400: '400',
        },
    ),
    html.H3('Update Announce Date'),
    dcc.Slider(
        id='announce_date',
        min=0,
        max=365,
        step=1,
        value=180,
        marks={i: str(i) for i in range(0, 366, 10)}
    ),
    html.H3('Update Student Body Size'),
    dcc.Slider(
        id='student_body_size',
        min=0,
        max=70000,
        step=100,
        value=10000,
        marks={i: str(i) for i in range(0, 70001, 10000)}
    )    
])

# Create callback for histogram using all the values provided and the update_prediction method below
@app.callback([Output('number-of-boosters', 'children'), Output('distribution-of-values', 'figure'), Output('map', 'figure')], 
                [Input('type', 'value'), Input('ranking', 'value'), Input('announce_date', 'value'), Input('student_body_size', 'value')])

# Note that Dash calls this method with default values when it starts.
def update_prediction(type, ranking, announce_date, student_body_size):    
    """
    Updates bar graph based on user-input. Note that in Sci-kit learn, the order of the columns matters. So, I need to transform my input.
    """
    # get all county fips from file NEED TO FIX ORDER IN EXECUTION
    # college_data = pd.read_csv('X_train_booster.csv')
    college_data = pd.read_csv('college_data_county.csv')
    college_data[['ranking', 'announce_date', 'Type', '2020.student.size']] = [ranking, announce_date, type, student_body_size]   
    college_data['ranking'] = pd.cut(college_data['ranking'], bins=[0, 20, 100, 200, 298, 400], labels=['a', 'b', 'c', 'd', 'e'], right=False)  # cut the ranking into 5 bins
    college_data_clean = college_data #.drop(columns=['state', 'state_new', 'STCOUNTYFP', 'state_fips', 'county_fips', 'county_fips_str', 'State', 'State Code', 'Division'])  
    college_data_clean['STCOUNTYFP'] = college_data_clean['STCOUNTYFP'].astype(str).str.zfill(5) # so map can read
    college_data_clean.drop(columns=['state', 'state_new', 'state_fips', 'county_fips', 'county_fips_str', 'State', 'State Code', 'Division'], 
            inplace=True)    
    # college_data_clean['2020.student.size'] = student_body_size # this is the last column for my sklearn features, so it also must be last here  
    college_data_clean['booster'] = model.predict(college_data_clean)
    print(college_data_clean)

    bar_fig = go.Figure()
    # create bar graph with bars for 0 and 1 with space between them
    bar_fig.add_trace(go.Bar(x=['0', '1'], y=[college_data_clean['booster'].value_counts()[0], college_data_clean['booster'].value_counts()[1]]))  

    # create map of US by county shaded based on output from fitted model
    # first load geojson file for US counties
    # All of the map stuff is copied from https://plotly.com/python/mapbox-county-choropleth/
    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response) 
    college_data_discrete = college_data_clean.copy()   
    college_data_discrete['booster'] = college_data_discrete['booster'] .astype('str') # so that a colormap doesn't show up--only 0 and 1
    map_fig = px.choropleth_mapbox(
        college_data_discrete, geojson=counties, locations='STCOUNTYFP', color='booster',        
        color_discrete_map={
            '0': '#F4EC15',
            '1': '#D95B43'
        },        
        mapbox_style="carto-positron",
        zoom=3, center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.5,
        labels={'STCOUNTYFP': 'County'}
    )
    map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return college_data_clean['booster'].sum(), bar_fig, map_fig

if __name__ == '__main__':
    app.run_server(debug=True)    

# Note: need to get list of counties and their data so that I don't have to call the api everytime. This can be accomplished by running cleaning.py; it's in the main method.
# Also, need to fix the order of the columns in the csv file.