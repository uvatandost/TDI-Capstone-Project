# TDI Capstone Project

## An interactive web app that visualizes conflict intensity and its relationship with government democracy qualities and predicts whether a social unrest could go violent based on underlying cause(s)

This project involves two sections. 

- First section aims to perform an exploratory data analysis and visualization on the dataset provided by The Quality of Governments Institute to explore the relationship between select government democracy qualities and conflict intensity (how serious social, ethnic and religious conflicts are on a scale of 1-10, 1 being there are no violent incidents and 10 being there is civil war or a widespread violent conflict based on social, ethnic or religious differences) in any given country in the dataset. The analysis reveals a strong relationship between those democracy qualities and conflict intensity. 

- The second section includes a machine learning model (random forest classifier) that predicts the probability that a social unrest could go violent based on underlying cause(s). The model was trained on a combination of two datasets one of which includes 5k data points of such events in Latin American countries and the other includes 15k data points of such events in African countries. Through feature engineering new features were created that depict the issues categories and were used as the predictor variables. And a target feature was created that includes two classes of events: peaceful and violent. The model classifies the events with over 80% accuracy. 

Link to the app: https://tdi-capstone-project.herokuapp.com/ 