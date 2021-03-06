import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from census import Census
import os
import us
import requests
import math
import nltk
import re
import string
from collections import Counter


def cleaning(covid_dates, date_cols=['Spring2020', 'FirstVaccine', 'Booster', 'Spring2022'], 
             census_vars={'B07011_001E': 'median_income', 'B01003_001E': 'total_population', 'B25010_003E': 'avg_hhsize'}, 
             last_tracking_date='4/2/2021',
             election_year=2016,
             scorecard_vars = {'size': '2020.student.size'},
             call_scoreboard_api = False,
             college_name='name',
             ignore_college=False,
             county_fips=False,
             skip_census=False):
    """  
    Given dataframe with zip code (and no state column) will find census data for given variables and corresponding county-level covid data, 
    as well as political leaning of both the county and state. If zip code not found, will set it to NaN.
    Returns the original dataframe with added columns. 
    Note that anything dataframe passed to this function may not be cleaned (i.e., have columns that make no sense or burdensome column names).
    So, may need additional cleaning for the dataframe's native columns.
    
    Arguments:
    covid_dates -- a dataframe with a column that holds the names of a college
                    and 'zip' with a valid zip code.
    date_cols -- list of column names that represent dates each college acted to impose a guideline.
    census_vars -- dictionary with codes and names of corresponding census variables from the US Census Bureau's ASC5 survey.
    last_tracking_date -- date before which to track all COVID data. Default is the day of my earliest recorded vaccination mandate.
    election_year -- year to get county-level data for presidential voting. Default is 2016 because 2020 not present in data yet.
    scorecard_vars -- dictionary with codes and names of corresponding variables from the US Dept of Education College Scorecard.
    call_scoreboard_api -- true if using a set of scorecard vars different from the default.
    college_name -- name of column in covid_dates holding college names.
    ignore_college -- true to ignore all data obtained from specific college. Must be inputed individually by user instead.
    county_fips -- true if using county fips data instead of zip codes. Column name in covid_dates must be 'STCOUNTYFP'.
    skip_census -- true if not calling the get_census method.
    """
    if date_cols is not None:
        covid_dates_only_d = covid_dates[date_cols].apply(pd.to_datetime) # ensure date columns in datetime format        
        first_dates = covid_dates_only_d.min()             
        date_diff = covid_dates_only_d - first_dates
        covid_dates[date_cols] = date_diff.apply(lambda x: x.dt.days)

    def get_census(covid_dates_cleaned, census_vars, county_fips):        
        census_api_key = os.getenv('api_key_census')
        c = Census(census_api_key, year=2020)
        state_fips = us.states.mapping('abbr', 'fips')
        if not county_fips:                               
            county_zips = pd.read_csv('zip-county-fips/ZIP-COUNTY-FIPS_2017-06.csv')
            if 'state' in covid_dates.columns:
                covid_dates_cleaned = covid_dates_cleaned.merge(county_zips[["ZIP", "STATE", "STCOUNTYFP"]], left_on=["zip"], right_on=["ZIP"]).drop(columns=["STATE", "ZIP"])
            else:
                covid_dates_cleaned = covid_dates_cleaned.merge(county_zips[["ZIP", "STATE", "STCOUNTYFP"]], left_on=["zip"], right_on=["ZIP"]).drop(columns=["ZIP"])
        else:
            covid_dates_cleaned = covid_dates
        covid_dates_cleaned.rename(columns={'STATE': 'state'}, inplace=True)                
        covid_dates_cleaned['state_fips'] = covid_dates_cleaned['state'].map(state_fips)                    
        covid_dates_cleaned["county_fips"] = covid_dates_cleaned["STCOUNTYFP"]%1000
        covid_dates_cleaned['county_fips_str'] = covid_dates_cleaned['county_fips'].astype(str).str.zfill(3)
        api_return_cols = list(census_vars.values()) + ['state', 'county']
        def separate_county_fips(x, *v):    
            """
            Given a row of data, will return the census data corresponding to census vars for the county and state.
            """
            api_return = c.acs5.state_county(v, x[0], x[1]) # returns dict in a list if found; empty list if not    
            try:
                api_return_clean = pd.Series(api_return[0])
                return api_return_clean.rename(census_vars) 
            except:
                no_api_return = pd.Series(index=api_return_cols, dtype='object')
                no_api_return[['state', 'county']] = x.values
                return no_api_return
        census_var_names = list(census_vars.keys())
        census_vars_counties = covid_dates_cleaned[['state_fips', 'county_fips_str']].drop_duplicates().apply(separate_county_fips, args=(census_var_names), axis=1)              
        covid_dates_cleaned = (covid_dates_cleaned.merge(census_vars_counties, 
                                                left_on=['state_fips', 'county_fips_str'], 
                                                right_on=['state', 'county'],
                                                suffixes=('', '_new'))
                                .drop(columns=['county']))
        return covid_dates_cleaned        

    def get_covid_county(covid_dates_cleaned, last_tracking_date):
        covid_county = pd.read_csv('counties.timeseries.csv')
        covid_county_present = covid_county.merge(covid_dates_cleaned['STCOUNTYFP'], left_on='fips', right_on='STCOUNTYFP', how='right')
        covid_county_present['date'] = pd.to_datetime(covid_county_present['date'])        
        covid_county_present = covid_county_present.query("date < @last_tracking_date").copy()
        covid_county_present[['communityLevels.canCommunityLevel']] = covid_county_present[['communityLevels.canCommunityLevel']].fillna(0)
        community_levels = covid_county_present.groupby('fips')['communityLevels.canCommunityLevel'].mean()
        covid_dates_cleaned = covid_dates_cleaned.merge(community_levels.rename('avg_community_level'), left_on='STCOUNTYFP', right_on='fips', how='left')
        return covid_dates_cleaned

    def get_political_lean(covid_dates_cleaned, election_year):
        political_control_state = {'DC': 'Dem', 'AL': 'Rep', 'AK': 'Rep', 'AZ': 'Rep', 'AR': 'Rep', 'CA': 'Dem', 'CO': 'Dem', 'CT': 'Dem', 'DE': 'Dem', 'FL': 'Rep', 'GA': 'Rep', 'HI': 'Dem', 'ID': 'Rep', 'IL': 'Dem', 'IN': 'Rep', 'IA': 'Rep', 'KS': 'Div', 'KY': 'Div', 'LA': 'Div', 'ME': 'Dem', 'MD': 'Div', 'MA': 'Div', 'MI': 'Div', 'MN': 'Div', 'MS': 'Rep', 'MO': 'Rep', 'MT': 'Div', 'NE': 'Rep', 'NV': 'Dem', 'NH': 'Div', 'NJ': 'Dem', 'NM': 'Dem', 'NY': 'Dem', 'NC': 'Div', 'ND': 'Rep', 'OH': 'Rep', 'OK': 'Rep', 'OR': 'Dem', 'PA': 'Div', 'RI': 'Dem', 'SC': 'Rep', 'SD': 'Rep', 'TN': 'Rep', 'TX': 'Rep', 'UT': 'Rep', 'VT': 'Div', 'VA': 'Dem', 'WA': 'Dem', 'WV': 'Rep', 'WI': 'Div', 'WY': 'Rep'}
        print(covid_dates_cleaned)
        covid_dates_cleaned['political_control_state'] = covid_dates_cleaned['state'].map(political_control_state)
        county_pres = pd.read_csv('political-data/countypres_2000-2020.csv')        
        county_pres = county_pres.query("year == @election_year")
        county_pres = county_pres.dropna(subset=['county_fips'])
        county_pres['county_fips'] = county_pres['county_fips'].astype(int, copy=False)
        county_pres['percentvote'] = county_pres['candidatevotes']/county_pres['totalvotes']
        county_pres = county_pres.query("(party == 'REPUBLICAN' or party == 'DEMOCRAT') and mode == 'TOTAL'")
        county_pres_percents = (county_pres.query("party == 'DEMOCRAT'")[['county_fips', 'percentvote']]
                                .merge(county_pres.query("party == 'REPUBLICAN'")[['county_fips', 'percentvote']], 
                                        on='county_fips', suffixes=('_D', '_R')))
        county_pres_percents['county_vote_diff'] = county_pres_percents['percentvote_D'] - county_pres_percents['percentvote_R']
        county_pres_percents['STCOUNTYFP'] = county_pres_percents['county_fips'].astype(str)
        covid_dates_cleaned['STCOUNTYFP'] = covid_dates_cleaned['STCOUNTYFP'].astype(str)
        covid_dates_all = covid_dates_cleaned.merge(county_pres_percents[['STCOUNTYFP', 'county_vote_diff']], on='STCOUNTYFP', how='left')
        return covid_dates_all

    def get_region(covid_dates_all):
        census_regions = pd.read_csv('https://raw.githubusercontent.com/cphalpert/census-regions/master/us%20census%20bureau%20regions%20and%20divisions.csv')
        covid_dates_all = (covid_dates_all.merge(census_regions, left_on='state', right_on='State Code', how='left'))                    
        return covid_dates_all

    def get_school_data(covid_dates_all, call_scoreboard_api, scorecard_vars):
        if call_scoreboard_api:
            api_key_scorecard = os.getenv('api_key_scorecard')
            readable_fields = ','.join(scorecard_vars.values())
            url = f'https://api.data.gov/ed/collegescorecard/v1/schools.json?fields=id,school.name,school.zip,{readable_fields}&api_key={api_key_scorecard}'
            r = requests.get(url)
            r = r.json()
            total_page_nums = r['metadata']['total']/r['metadata']['per_page']
            scoreboard_data = []
            for page_num in range(math.ceil(total_page_nums)):
                url = f'https://api.data.gov/ed/collegescorecard/v1/schools.csv?fields=id,school.name,school.zip,2020.student.size&page={page_num}&api_key={api_key_scorecard}'
                scoreboard_data.append(pd.read_csv(url))
            scoreboard_data_all = pd.concat(scoreboard_data)
            scoreboard_data_all.reset_index(drop=True, inplace=True)

            # clean zip (disregard last 4 digits)
            scoreboard_data_all['school.zip'] = scoreboard_data_all['school.zip'].astype(str)
            scoreboard_data_all['school.zip'] = scoreboard_data_all['school.zip'].str.extract(r'^(\d+)')
            def separate_connected_zips(x):
                """
                Get first 5 digits of zip if full zip exists and is not separated by a '-'.
                """    
                if len(x) == 9:
                    return x[:5]
                return x.lstrip('0') # because all the previous zip codes I've worked with also ignored leading 0s
            scoreboard_data_all['school.zip'] = scoreboard_data_all['school.zip'].apply(separate_connected_zips)
        else:
            scoreboard_data_all = pd.read_csv('scoreboard_size.csv')
            scoreboard_data_all['school.zip'] = scoreboard_data_all['school.zip'].astype(str)


        covid_dates_all['zip_str'] = covid_dates_all['zip'].astype(str) # need str to do merging
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(['university', 'college', 'main', 'campus'])
        def clean_name(x, is_list=False):
            text = re.sub('-', ' ', x) # dashes in data -- need to add spaces and remove
            text = text.split(' ') # remove excess spaces
            text = ' '.join(text)
            text = ''.join([i.lower() for i in text if i not in string.punctuation])
            text = nltk.tokenize.word_tokenize(text)    
            text = [i for i in text if i not in stopwords] # lower all
            text = text if is_list else ' '.join(text)    
            return text
        covid_dates_all['cleaned_name_list'] = covid_dates_all[college_name].apply(lambda x: clean_name(x, True))
        scoreboard_data_all['cleaned_school.name_list'] = scoreboard_data_all['school.name'].apply(lambda x: clean_name(x, True))
        # merge on zips and partially on names (keep name in scoreboard data that's closest in cosine similarity to the one in covid_dates_all)
        covid_dates_all_zips = covid_dates_all.merge(scoreboard_data_all, left_on='zip_str', right_on='school.zip', how='left')
        def cosine_similarity(l1, l2):
            """
            Copied from Martijn Pieters's answer in 
            https://stackoverflow.com/questions/14720324/compute-the-similarity-between-two-lists
            """
            c1 = Counter(l1)
            c2 = Counter(l2)
            terms = set(c1).union(c2)
            dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
            magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
            magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
            return dotprod / (magA * magB)
        old_num = covid_dates_all_zips[college_name].unique().shape[0]
        covid_dates_all_zips.dropna(subset=['2020.student.size'], inplace=True)
        new_num = covid_dates_all_zips[college_name].unique().shape[0]
        perc_dropped = 1 - new_num/old_num
        print(f'Note that {perc_dropped*100}% of samples have been dropped due to automated college searching for the college scoreboard.')
        covid_dates_all_zips['name_similarity'] = covid_dates_all_zips.apply(lambda x: cosine_similarity(x['cleaned_name_list'], x['cleaned_school.name_list']), axis=1)
        covid_dates_all = covid_dates_all_zips.loc[covid_dates_all_zips.groupby(college_name)['name_similarity'].idxmax().values] # keep only college in zip code with highest similarty
        return covid_dates_all

    # try:
    if not skip_census:
        covid_dates = get_census(covid_dates, census_vars, county_fips)    
    covid_dates = get_covid_county(covid_dates, last_tracking_date)    
    covid_dates = get_political_lean(covid_dates, election_year)    
    covid_dates = get_region(covid_dates)
    if not ignore_college:
        covid_dates = get_school_data(covid_dates, call_scoreboard_api, scorecard_vars)
    return covid_dates
    # except:
    #     print('All zips not found.')
    #     return covid_dates

if __name__ == '__main__':
    # use to test the dataset I created
    # covid_dates = pd.read_csv('covid_dates_nice.csv') # after basic cleaning applied to my excel file    
    # covid_dates = cleaning(covid_dates)    
    # covid_dates.to_csv('covid_dates_cleaned_script_school.csv', index=False) 

    # use for normal vaccination status inquiries
    # covid_dates = pd.read_csv('vacc_mandates_top.csv') 
    # covid_dates = cleaning(covid_dates, date_cols=['announce_date'], last_tracking_date='3/25/2021', college_name='College')    
    # covid_dates.to_csv('vacc_mandates_cleaned_school.csv', index=False)

    # use for boosters -- changed last tracking date to the earliest booster day I found in my dataset because the other set only tracks vaccine mandates
    # covid_dates = pd.read_csv('vacc_mandates_top.csv')   
    # covid_dates = cleaning(covid_dates, date_cols=['announce_date'], last_tracking_date='12/06/2021', college_name='College')    
    # covid_dates.to_csv('booster_mandates_cleaned_school.csv', index=False)    
        
    # Get county-level values for all counties--can't do all at once because of API limits
    # county_zips = pd.read_csv('zip-county-fips/ZIP-COUNTY-FIPS_2017-06.csv')
    # n = 100 # number of times to call API
    # college_data = county_zips[['STCOUNTYFP', 'STATE']].drop_duplicates() #.head(n)
    
    # Get all counties from Census API and add necessary columns to run with method
    # census_api_key = os.getenv('api_key_census')
    # c = Census(census_api_key, year=2020)
    # census_vars={'B07011_001E': 'median_income', 'B01003_001E': 'total_population', 'B25010_003E': 'avg_hhsize'}
    # census_var_names = list(census_vars.keys())
    # api_return = c.acs5.state_county(census_var_names, Census.ALL, Census.ALL)
    # api_return_clean = pd.DataFrame(api_return)    
    # api_return_clean = api_return_clean.rename(columns=census_vars)
    # api_return_clean = api_return_clean.rename(columns={'state': 'state_fips', 'county': 'county_fips_str'})
    # state_fips = us.states.mapping('fips', 'abbr')
    # api_return_clean['state'] = api_return_clean['state_fips'].map(state_fips)    
    # api_return_clean['STCOUNTYFP'] = (api_return_clean['state_fips'] + api_return_clean['county_fips_str']).astype(int)      
    # # Get remaining data  
    # last_tracking_date = '12/06/2021' # for booster
    # college_data = cleaning(api_return_clean, date_cols=None, last_tracking_date=last_tracking_date, ignore_college=True, county_fips=True, skip_census=True)
    # college_data.to_csv('college_data_county.csv', index=False)
    
    import joblib
    ranking = 1
    announce_date = 2
    type = 'Public'
    student_body_size = 2000
    model = joblib.load('booster_model_jlib')
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
    print(college_data_clean.drop(columns='STCOUNTYFP').dropna()) # drop rows if there are NaN values in any columns
    college_data_clean['booster'] = model.predict(college_data_clean.drop(columns='STCOUNTYFP').dropna())
