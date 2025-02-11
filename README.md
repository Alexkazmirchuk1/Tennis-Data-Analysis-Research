# Dynamic Tennis Prediction Model

## Overview
#### This project simulates a tennis match from start to end and assess at which points we can confidently forecast the result. We use data from the 2021 and 2023 Wimbledon Tournaments to train and evaluate our model.

---
![preview](ex.visual.png)

#### Players are assumed to begin with an equal liklihood of victory, with such likelihood dynamically updating throughout the match as events occur. Factors such as server success, unforced errors, and ace points are measured to effect the swing of momentum in real time. 

#### In the 2023 Wimbledon Final between Alcaraz and Djokovic we witness a hard fought bout capped off with an impressive comeback from Alcaraz. Djokovic looked to take an early and dominant lead in this match, but Alcaraz's strong play propelled him into contention by the third set. After that it was a back and forth contest until the final point was played. We analyze which events during this match had the greatest effect on the eventual outcome, and how we can generalize this approach to accurately forecast any given tennis match.

## Requirments
* numpy
* pandas
* seaborn
* matplotlib

## Citations
Wilkens, Sascha. "Sports prediction and betting models in the machine learning age: The case of tennis." Journal of Sports Analytics 7.2 (2021): 99-117.

Carrari, Andrea, Marco Ferrante, and Giovanni Fonseca. "A new Markovian model for tennis matches." Electronic Journal of Applied Statistical Analysis 10.3 (2017): 693-711.

Yue, Jack C., Elizabeth P. Chou, Ming-Hui Hsieh, and Li-Chen Hsiao. "A study of forecasting tennis matches via the Glicko model." PLOS ONE 17.4 (2022): e0266838.

ATP Tour. "Roger Federer: Player Stats: ATP Tour: Tennis." 2024. Accessed October 9, 2024. https://www.atptour.com/en/players/roger-federer/f324/player-stats?year=all&surface=all.

Sackmann, Jeff. "Tennis Slam Point-by-Point Data." 2024. Accessed November 17, 2024. https://github.com/JeffSackmann/tennis_slam_pointbypoint.

Clarke, S. R., and D. Dyte. "Using official ratings to simulate major tennis tournaments." International Transactions in Operational Research 7.6 (2000): 585-594. https://doi.org/10.1111/j.1475-3995.2000.tb00218.x.

Independent. "Carlos Alcaraz wins Wimbledon 2023: Spanish star claims his first Wimbledon title." The Independent, 2023. https://www.independent.co.uk/sport/tennis/wimbledon-final-2023-winner-trophy-carlos-alcaraz-b2376360.html. Accessed December 5, 2024.
