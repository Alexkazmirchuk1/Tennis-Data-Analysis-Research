# Dynamic Tennis Prediction Model

## Overview
#### This project simulates a tennis match from start to end and asses at which points we can confidently forecast the result. We use data from the 2021 and 2023 Wimbledon Tournaments to inform and assess our model.

---
![preview](ex.visual.png)

#### Players are assumed to begin with an equal liklihood of victory, with such likelihood dynamically updating throughout the match as events occur. Factors such as server success, unforced errors, and ace points are measured to effect the swing of momentum in real time. 

#### In the 2023 Wimbledon Final between Alcaraz and Djokovic we witness a hard fought bout capped off with an impressive comeback from Alcaraz. Djokovic looked to take an early and dominant lead in this match, but Alcaraz's strong play propelled him into contention by the third set. After that it was a back and forth contest until the final point was played. We analyze which events during this match had the greatest effect on the eventual outcome, and how we can generalize this approach to accurately forecast any given tennis match.

## Requirments
* numpy
* pandas
* seaborn
* matplotlib
