# Note: using Github Copilot for the first time in a file.
# Also using https://towardsdatascience.com/build-a-machine-learning-simulation-tool-with-dash-b3f6fd512ad6 heavily as a guide.
# Thanks to the author Pierre-Louis Bescond. Check out the article for more information.
# Using https://towardsdatascience.com/3-easy-ways-to-make-your-dash-application-look-better-3e4cfefaf772 for styling advice.

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.express as px

# import fitted model
import joblib
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
            [html.P('This app predicts if a college in each county with the given parameters will have a \
                booster mandate, given a vaccine mandate is in place. Note that it will not be entirely accurate \
                as the mandates are difficult to predict, but should give a general idea of the trends.'),
            ]
        ),
        style={'margin-left': '50px', 'margin-right': '50px'}
    ),
    dbc.Row(
        dbc.Col(            
            dcc.Graph(id='map')       
        )
    ),    
    dbc.Row(
        [dbc.Col('Type of School'),
        dbc.Col('Ranking'),
        dbc.Col('Announce Date of Vacc Mandate'),
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
                value='Private',
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
    dbc.Row(
        html.H3('Percent of Counties with Booster Mandates'),
        style={'margin-top': '50px'},
    ),
    dbc.Row(
        [html.H4(id='number-of-boosters'),  
         html.P('with probability cutoff 0.5')]
    ),
    dbc.Row(
        dcc.Graph(id='distribution-of-values'),
    ),
    dbc.Row(
        html.A('Jupyter notebook with details of model creation', href='https://github.com/ncrispino/covid_university_dates/blob/master/Covid%20Booster%20Model.ipynb')
    ) 
])

# Create callback for histogram using all the values provided and the update_prediction method below
@app.callback([Output('number-of-boosters', 'children'), Output('distribution-of-values', 'figure'), Output('map', 'figure')], 
                [Input('type', 'value'), Input('ranking', 'value'), Input('announce_date', 'value'), Input('student_body_size', 'value')])

# Note that Dash calls this method with default values when it starts.
def update_prediction(type, ranking, announce_date, student_body_size):    
    """
    Updates data and figures for all counties based on user-input. Note that in Sci-kit learn, the order of the columns matters, so I have to do some preprocessing.
    """
    # get data for all counties and merge with user-selected values 
    college_data = pd.read_csv('college_data_county.csv')
    column_names = college_data.columns
    college_data[['ranking', 'announce_date', 'Type']] = [ranking, announce_date, type]
    college_data = college_data[['ranking', 'announce_date', 'Type', *column_names]]
    college_data['ranking'] = pd.cut(college_data['ranking'], bins=[0, 20, 100, 200, 298, 400], labels=['a', 'b', 'c', 'd', 'e'], right=False)  # cut the ranking into 5 bins
    college_data_clean = college_data #.drop(columns=['state', 'state_new', 'STCOUNTYFP', 'state_fips', 'county_fips', 'county_fips_str', 'State', 'State Code', 'Division'])  
    college_data_clean['STCOUNTYFP_int'] = college_data_clean['STCOUNTYFP']
    college_data_clean['STCOUNTYFP'] = college_data_clean['STCOUNTYFP'].astype(str).str.zfill(5) # so map can read
    college_data_clean.drop(columns=['state', 'state_fips', 'county_fips_str', 'State Code', 'Division'], 
            inplace=True)    
    college_data_clean['2020.student.size'] = student_body_size # this is the last column for my sklearn features, so it also must be last here          
    college_data_booster = model.predict(college_data_clean.drop(columns=['STCOUNTYFP', 'State']).dropna())
    college_data_booster_proba = model.predict_proba(college_data_clean.drop(columns='STCOUNTYFP').dropna())            
    college_data_clean = college_data_clean.join(pd.Series(college_data_booster, name='booster'), how='right')
    college_data_clean = college_data_clean.join(pd.DataFrame(college_data_booster_proba, columns=['0', 'Booster Probability']).drop(columns=['0']), how='right')
    num_boosters = str(college_data_clean['booster'].sum()/college_data_clean.shape[0]*100) + '%'
    county_names = pd.read_csv('https://github.com/kjhealy/fips-codes/blob/master/state_and_county_fips_master.csv?raw=true').drop(columns=['state'])    
    college_data_clean = college_data_clean.merge(county_names, left_on='STCOUNTYFP_int', right_on='fips', how='left')

    # create histogram of booster probabilities
    hist_fig = go.Figure(data=[go.Histogram(
        x=college_data_clean['Booster Probability'],
        name='Booster Probability',
        marker_color='#3D9970'
    )])
    hist_fig.add_vline(x=0.5, line_dash='dash')
    hist_fig.update_layout(xaxis_title_text='Booster Probability', yaxis_title_text='Count')
    hist_fig.update_layout(title_x=0.5, title_y=0.9)
    hist_fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
    hist_fig.update_layout(xaxis=dict(
        title='Booster Probability',
    ))
    hist_fig.update_layout(yaxis=dict(
        title='Count',
    ))
    hist_fig.update_layout(autosize=False)        

    # create map of US by county shaded based on output from fitted model
    # first load geojson file for US counties
    # All of the map stuff is copied from https://plotly.com/python/mapbox-county-choropleth/
    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response) 
    map_fig = px.choropleth_mapbox(
        college_data_clean, geojson=counties, locations='STCOUNTYFP', color='Booster Probability',    
        color_continuous_scale=px.colors.sequential.Agsunset,        
        color_continuous_midpoint=0.5,              
        mapbox_style="carto-positron",
        zoom=3, center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.5,
        labels={'STCOUNTYFP': 'County'},
        hover_name='name',
        hover_data=['State', 'STCOUNTYFP', 'Booster Probability']
    )
    map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, legend={'orientation': 'h', 'y': 1, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    return num_boosters, hist_fig, map_fig

if __name__ == '__main__':
    app.run_server(debug=True)