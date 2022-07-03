# Note: using Github Copilot for the first time in a file.
# Also using https://towardsdatascience.com/build-a-machine-learning-simulation-tool-with-dash-b3f6fd512ad6 heavily as a guide.
# Thanks to the author Pierre-Louis Bescond. Check out the article for more information.
# Using https://towardsdatascience.com/3-easy-ways-to-make-your-dash-application-look-better-3e4cfefaf772 for styling advice.

from distutils.log import debug
from re import M
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.figure_factory as ff
import plotly.express as px

# import fitted model
# from sklearn.externals import joblib
import joblib
# model = joblib.load('booster_dummy_model_jlib')
# model = joblib.load('booster_log_model_jlib')
model = joblib.load('booster_model.joblib')

# Create dash app
app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
load_figure_template('LUX')

# Page structure will be:
    # Map of US by county shaded based on output from fitted model
    # Bar graph of output from fitted model
    # Total number of booster mandates
    # Slider for type of school, with two options
    # Slider to update ranking with 5 bins of range bins=[0, 20, 100, 200, 298, 400]
    # Slider to update announce_date
    # Slider to update student body size

# apply basic HTML layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '90%', 'margin-left': '50px', 
'margin-right': '50px', 'margin-top': '25px', 'margin-bottom': '25px'},
children=[
    dbc.Row(
        dbc.Col(
            html.H1('Booster Mandate Predictor'),
        )
    ),
    dbc.Row(
        dbc.Col(
            # show map from callback
            dcc.Graph(id='map'),
            # style = {'margin-left': '25px', 'margin-right': '25px'}
        )
    ),
    # make note that says that counties that are not shaded did not have enough data to run the model
    dbc.Row(
        [dbc.Col('Type of School'),
        dbc.Col('Ranking'),
        dbc.Col('Announce Date'),
        dbc.Col('Student Body Size')],
        style={'text-decoration': 'underline'}
    ),
    dbc.Row(
        [dbc.Col(
            [dcc.RadioItems(
                id='type',
                options=[
                    {'label': 'Public', 'value': 'Public'},
                    {'label': 'Private', 'value': 'Private'},
                ],
                value='Public',
                labelStyle={'display': 'inline-block'},
                inputStyle={'margin-left': '10px', 'margin-right': '10px'},
                )]
        ),
        dbc.Col(                      
            [dcc.Slider(
                id='ranking',
                min=0,
                max=4,
                step=1,
                value=3,
                marks={
                    0: '0-19',
                    1: '20-99',
                    2: '100-99',
                    3: '200-98',
                    4: '299+'                    
                }
            )]
        ),
        dbc.Col(            
            [dcc.Slider(
                id='announce_date',
                min=0,
                max=200,
                step=1,
                value=20,
                marks={i: str(i) for i in range(0, 201, 50)}
            )]
        ),
        dbc.Col(            
            [dcc.Slider(
                id='student_body_size',
                min=0,
                max=70000,
                step=100,
                value=10000,
                marks={i: str(i//1000) + 'k' for i in range(0, 70001, 10000)}
            )]
        )]
    ),
    dbc.Row(
        html.P('Counties that are not shaded did not have enough data to run the model', style={'font-size': '12px', 'font-style': 'italic'}),
        style={'margin-top': '10px'},
    ),
    # html.H3('Distribution of Values'),
    # show the number of boosters from callback
    html.H3('Number of Boosters'),
    html.H4(id='number-of-boosters'),    
    dcc.Graph(id='distribution-of-values'),  
])

# Create callback for histogram using all the values provided and the update_prediction method below
@app.callback([Output('number-of-boosters', 'children'), Output('distribution-of-values', 'figure'), Output('map', 'figure')], 
                [Input('type', 'value'), Input('ranking', 'value'), Input('announce_date', 'value'), Input('student_body_size', 'value')])

# Note that Dash calls this method with default values when it starts.
def update_prediction(type, ranking, announce_date, student_body_size):    
    """
    Updates bar graph based on user-input. Note that in Sci-kit learn, the order of the columns matters. So, I need to transform my input.
    """
    # get data for all counties and merge with user-selected values 
    college_data = pd.read_csv('college_data_county.csv')
    column_names = college_data.columns
    college_data[['ranking', 'announce_date', 'Type']] = [ranking, announce_date, type]
    college_data = college_data[['ranking', 'announce_date', 'Type', *column_names]]
    college_data['ranking'] = pd.cut(college_data['ranking'], bins=[0, 20, 100, 200, 298, 400], labels=['a', 'b', 'c', 'd', 'e'], right=False)  # cut the ranking into 5 bins
    college_data_clean = college_data #.drop(columns=['state', 'state_new', 'STCOUNTYFP', 'state_fips', 'county_fips', 'county_fips_str', 'State', 'State Code', 'Division'])  
    college_data_clean['STCOUNTYFP'] = college_data_clean['STCOUNTYFP'].astype(str).str.zfill(5) # so map can read
    college_data_clean.drop(columns=['state', 'state_fips', 'county_fips_str', 'State', 'State Code', 'Division'], 
            inplace=True)    
    college_data_clean['2020.student.size'] = student_body_size # this is the last column for my sklearn features, so it also must be last here          
    college_data_booster = model.predict(college_data_clean.drop(columns='STCOUNTYFP').dropna())      
    college_data_clean = college_data_clean.join(pd.Series(college_data_booster, name='booster'), how='right')    

    bar_fig = go.Figure()
    # create bar graph with bars for 0 and 1 with space between them
    booster_counts = college_data_clean['booster'].value_counts().values
    if len(booster_counts) == 1:
        booster_counts = np.append(booster_counts, 0)
    bar_fig.add_trace(go.Bar(x=['0', '1'], y=[booster_counts[0], booster_counts[1]]))  

    # create map of US by county shaded based on output from fitted model
    # first load geojson file for US counties
    # All of the map stuff is copied from https://plotly.com/python/mapbox-county-choropleth/
    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response) 
    college_data_discrete = college_data_clean.copy()   
    college_data_discrete['booster'] = college_data_discrete['booster'].astype('str') # so that a colormap doesn't show up--only 0 and 1
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
    map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, legend={'orientation': 'h', 'y': 1, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    return college_data_clean['booster'].sum(), bar_fig, map_fig

if __name__ == '__main__':
    app.run_server(debug=True)    

# Note: need to get list of counties and their data so that I don't have to call the api everytime. This can be accomplished by running cleaning.py; it's in the main method.
# Also, need to fix the order of the columns in the csv file.