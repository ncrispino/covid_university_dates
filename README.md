# covid_university_dates
Predicting aspects of university COVID guidelines using machine learning.

- Data cleaning is done in 'Analyzing Covid Decision Dates' with my own data. A generalized script is in cleaning.py, which is applied to the vaccine data in 'Vaccine Mandates.'
- Preprocessing and model creation is done in 'Covid Booster Model.' Note that 'Covid Model Creation' was my first attempt at model creation where I used my own data, but it is unfinished, as I decided to use other data in the end.
- Dash app in 'dash_model.py'; deployed [here](https://covid-university-boosters.herokuapp.com/).

<details><summary>I also attempted to track specific actions made by universities in covid_dates_creation</summary><br/>

Uses [Uni-rank](https://github.com/nahid18/uni-rank) package to get baseline data, though I had to fix it in 'Documenting COVID Decisions at US Universities'. 

Columns with the dates of COVID-related decisions were created by me.

Most of the dates should be correct (or at least within a few days of being correct); however, let me know if I've made a mistake, or am missing data you have.

Use "cleaned_university_covid_dates" for any projects, though you probably want to clean the code more, as done in my machine learning prep files.
</details>
