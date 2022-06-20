import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def cleaning(covid_dates, date_cols=['Spring2020', 'FirstVaccine', 'Booster', 'Spring2022'], 
             census_vars={'B07011_001E': 'median_income', 'B01003_001E': 'total_population', 'B25010_003E': 'avg_hhsize'}, 
             last_tracking_date='4/2/2021',
             election_year=2016):
    """  
    Given dataframe with state and zip code will find census data for given variables and corresponding county-level covid data, 
    as well as political leaning of both the county and state. 
    Returns the original dataframe with added columns. 
    Note that anything dataframe passed to this function may not be cleaned (i.e., have columns that make no sense or burdensome column names).
    So, may need additional cleaning for the dataframe's native columns.
    
    Arguments:
    covid_dates -- a dataframe with a column called 'state', with the two letter state abbreviation, and 'zip' with a valid zip code.
    date_cols -- list of column names that represent dates each college acted to impose a guideline.
    census_vars -- dictionary with codes and names of corresponding census variables from the ASC5 survey.
    last_tracking_date -- date before which to track all COVID data. Default is the day of my earliest recorded vaccination mandate.
    election_year -- year to get county-level data for presidential voting. Default is 2016 because 2020 not present in data yet.
    """
    if date_cols is not None:
        covid_dates_only_d = covid_dates[date_cols].apply(pd.to_datetime) # ensure date columns in datetime format        
        first_dates = covid_dates_only_d.min()             
        date_diff = covid_dates_only_d - first_dates
        covid_dates[date_cols] = date_diff.apply(lambda x: x.dt.days)

    def get_census(covid_dates_cleaned, census_vars):
        from census import Census
        import os
        import us
        census_api_key = os.getenv('api_key_census')
        c = Census(census_api_key, year=2020)
        state_fips = us.states.mapping('abbr', 'fips')
        covid_dates_cleaned['state_fips'] = covid_dates_cleaned['state'].apply(lambda x: state_fips[x])  
        county_zips = pd.read_csv('zip-county-fips/ZIP-COUNTY-FIPS_2017-06.csv')
        covid_dates_cleaned = covid_dates_cleaned.merge(county_zips[["ZIP", "STATE", "STCOUNTYFP"]], left_on=["state", "zip"], right_on=["STATE", "ZIP"]).drop(columns=["ZIP", "STATE"])
        covid_dates_cleaned["county_fips"] = covid_dates_cleaned["STCOUNTYFP"]%1000
        covid_dates_cleaned['county_fips_str'] = covid_dates_cleaned['county_fips'].astype(str).str.zfill(3)
        api_return_cols = list(census_vars.values()) + ['state', 'county']
        def separate_county_fips(x, *v):    
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
                                .drop(columns=['state_new', 'county']))
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
    
    covid_dates = get_census(covid_dates, census_vars)    
    covid_dates = get_covid_county(covid_dates, last_tracking_date)    
    covid_dates = get_political_lean(covid_dates, election_year)    
    return covid_dates

if __name__ == '__main__':
    covid_dates = pd.read_csv('covid_dates_nice.csv') # after basic cleaning applied to my excel file    
    covid_dates = cleaning(covid_dates)    
    covid_dates.to_csv('covid_dates_cleaned_script.csv', index=False)