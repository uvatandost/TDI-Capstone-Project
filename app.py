import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import dill
"""
# How Do a Government's Democracy Qualities Influence Internal Conflicts in a Country?
"""
"\n",
"\n",
"\n",
st.write("Conflicts and wars seem to never end in the world we're livig in. This project is to perform an exploratory data analysis and visualization on the dataset provided by The Quality of Governments Institute to explore the relationship between select government democracy qualities and conflict intensity (how serious social, ethnic and religious conflicts are on a scale of 1-10, 1 being there are no violent incidents and 10 being there is civil war or a widespread violent conflict based on social, ethnic or religious differences) in any given country. The analysis reveals a strong relationship between those democracy qualities and conflict intensity which helps explain why conflict is very common in some particular regions.")


# Loading QoG data-----------------------------------------------------------
data_source = "qog_cleaned.csv"


qog = pd.read_csv(data_source)


#qog2 = qog1.dropna()
qog.columns = ['Country Name','year', 'Conflict Intensity', 'Socio-Economic Barriers', 'State Identity',
	'Separation of Powers', 'Rule of Law', 'Political Participation',
	'Performance of Democratic Institution', 'Independant Judiciary',
	'Freedom of Expression', 'Free and Fair Elections', 'Equal Opportunity',
	'Civil Society Participation', 'Civil Rights',
	'Association/Assembly Rights', 'ccodealp']

# Year slider
st.sidebar.write("\n")
year = st.sidebar.slider(label = 'Slide to change the year', min_value=2005, max_value=2019,
	 value=2005, step=2, format='%d', key=None, help=None)

country = st.sidebar.text_input('Enter a Country:', 'Albania')


country = qog.loc[qog['Country Name']==country] # grouping data by country
country.set_index("year", inplace=True)
country = country.drop(["Country Name",'ccodealp'], axis=1)


df_year = qog.loc[qog["year"]==year] # grouping by year
df_year.set_index("Country Name", inplace=True)
df_year.drop('year', axis=1)

groupedby_year = qog.loc[qog.groupby('year').groups[year]]

"Please use the slider on the left to see how the map changes over the years."
# World Map matplotlib
df = groupedby_year

fig = go.Figure(data=go.Choropleth(
    locations = df['ccodealp'],
    z = df['Conflict Intensity'],
    text = df['Country Name'],
    colorscale = 'Reds',
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '',
    colorbar_title = 'Conflict Intensity',
))

fig.update_layout(width=800, height=500,
    title_text='Conflict Intensity',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='Source: <a href="https://www.gu.se/en/quality-government/qog-data/data-downloads/standard-dataset">\
            The QoG Institute</a>',
        showarrow = False
    )]
)

st.write(fig)


# Checkbox ------------------------------------------------------
bti_ci = st.sidebar.checkbox('Conflict Intensity')
bti_seb = st.sidebar.checkbox('Socio-Economic Barriers')
bti_si = st.sidebar.checkbox('State Identity')
bti_sp = st.sidebar.checkbox('Separation of Powers')
country1 = country

my_list = []
if bti_ci:
	my_list.append('Conflict Intensity')
if bti_seb:
	my_list.append('Socio-Economic Barriers')
if bti_si:
	my_list.append('State Identity')
if bti_sp:
	my_list.append('Separation of Powers')
if not my_list:
	my_list.append('Conflict Intensity')


country1 = country[my_list]


"In the expander below, you can see the original dataset along with a subset of it for a country you select on the left side panel."

with st.beta_expander("See the data"):
		"QoG Data",
		qog,
		"\n",
		"\n",
		"Subset of data for selected country"
		country
		"\n",
		"\n",    

"\n"
"\n"
"\n"
"Here you can see changes in any of the government qualities for a country over the years 2005 through 2019 for the country and its government qualities selected on the leftside panel."
st.line_chart(country1)



"\n"
# Multiselect options
options = st.sidebar.multiselect(
	 'Pick measures',
	 ['Socio-Economic Barriers', 'State Identity',
	'Separation of Powers', 'Rule of Law', 'Political Participation',
	'Performance of Democratic Institution', 'Independant Judiciary',
	'Freedom of Expression', 'Free and Fair Elections', 'Equal Opportunity',
	'Civil Society Participation', 'Civil Rights',
	'Association/Assembly Rights'])

"Finally, you can pick a measure from the panel on the left to see scatterplots below that illustrate the relationship between the selected government quality and conflict intensity across the countries. You can also use the year slidebar to change the year to see the trend over the years."

with st.beta_expander("See the data"):
		df_year

"\n"
"\n"
"\n"
# Altair Line scatterplots----------------------------------------

for item in options:
	if not item:
		item = 'Socio-Economic Barriers'
	chart = alt.Chart(df_year,width=600, height=350).mark_point(size=100).encode(
	    x=item,
	    y='Conflict Intensity:Q'
	)

	st.write((chart + chart.transform_regression(item, 'Conflict Intensity').mark_line(size=4)).interactive())


# Predictive modeling

with open('fitted_model.pkl', 'rb') as f:
	model = dill.load(f)

st.write('## Predictive Model')
st.write('The machine learning model can predict probability that a social unrest could go violent based on the underlying cause(s)')
options1 = st.multiselect(
	 'Please select at least one underlying issue for the social unrest to generate violance probability',
	 ['elections','economy, jobs', 
             'food, water, subsistence', 'environmental degradation','ethnic discrimination, ethnic issues',
             'religious discrimination, religious issues','education','foreign affairs/relations', 
             'domestic war, violence, terrorism', 'human rights, democracy', 'pro-government', 'economic resources/assets',
             'other','unknown, not-specified'])

underlying_issue = ['elections','economy, jobs', 
             'food, water, subsistence', 'environmental degradation','ethnic discrimination, ethnic issues',
             'religious discrimination, religious issues','education','foreign affairs/relations', 
             'domestic war, violence, terrorism', 'human rights, democracy', 'pro-government', 'economic resources/assets',
             'other','unknown, not-specified']

input_list = []
for item in underlying_issue:
	if item in options1:
		input_list.append(1)
	else:
		input_list.append(0)


probs = model.predict_proba([input_list])
prob_nonviol, prob_viol = round(probs[0,0],2), round(probs[0,1], 2)
st.write('Probability that the event will be peaceful:', prob_nonviol)
st.write('Probability that the event will go violent:', prob_viol)


fig = go.Figure(go.Bar(
            x=[prob_viol, prob_nonviol],
            y=['Violent', 'Peaceful'],
            orientation='h'))

st.plotly_chart(fig)

"*Author: Umit Vatandost*"