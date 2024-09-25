import numpy as np

class MatchStats:
    def __init__(self, raw_data, match_to_examine):
        
        self.match = raw_data[raw_data['match_id'] == match_to_examine]
        self.player1_name = self.match['player1'].values[0]
        self.player2_name = self.match['player2'].values[0]
        self.player1_surname = self.match['p1_lastname'].values[0]
        self.player2_surname = self.match['p2_lastname'].values[0]
        
        self.names = [self.player1_name, self.player2_name]
        self.surnames = [self.player1_surname, self.player2_surname]
        
        self.set_change_points = np.where( np.diff(self.match['set_no']) > 0 )[0]
        self.set_change_points = np.array([*self.set_change_points, self.match.shape[0]-1])
        
        # integer 1/2 of which player won final set. winner of final set 
        # must be the match winner.
        self.match_winner = self.match['set_victor'].iloc[-1]
        
        self.set_victors = self.match['set_victor'][self.match['set_victor'] != 0]
        self.point_victors = self.match['point_victor'][self.match['point_victor'] != 0]
        self.game_victors = self.match['game_victor'].values
        self.unf_err = 2*self.match['p2_unf_err'].values + self.match['p1_unf_err'].values        
        self.winner_id = self.match_winner
        self.winner_name = self.names[self.winner_id - 1]
        return
#

class SetWinnerModel:
    '''
    At the end of set i, whoever won *that set* is predicted to be the winner
    of the overall match.
    '''
    def __init__(self, raw_data, match_to_examine):
        # import the calculation of these match stats variables into this class.
        MatchStats.__init__(self,raw_data,match_to_examine)
        return
    
    def fit(self):
        return # nothing to process with the time series.
    
    def prediction(self):
        '''
        Output: prediction at the end of set 1,2,3,4,5 (if applicable) 
        to predict the winner of the match.
        '''
        pred = self.match['set_victor'][ self.match['set_victor']!=0 ].values
        pred = pred
        return pred


class CumulativeSetWinnerModel:
    '''
    At the end of set i, whoever won *more sets until that point* is predicted
    to be winner. 
    
    If it is a tie, whoever had previously been in the lead is marked as the 
    prediction.
    '''
    def __init__(self, raw_data, match_to_examine):
        # import the calculation of these match stats variables into this class.
        MatchStats.__init__(self,raw_data,match_to_examine)
        return
    
    def fit(self):
        self.p1_cumulative = np.cumsum( self.set_victors.values == 1 )
        self.p2_cumulative = np.cumsum( self.set_victors.values == 2 )
        return
    
    def prediction(self):
        '''
        Output: prediction at the end of set 1,2,3,4,5 (if applicable) 
        to predict the winner of the match.
        '''
        pred = (self.p2_cumulative - self.p1_cumulative) > 0
        pred = pred.astype(int) + 1
        for i,p in enumerate(pred):
            if p==0:
                pred[i] = pred[i-1]
        return pred

# TODO
#class CumulativePointWinnerModel:
class CumulativePointWinnerModel:
    '''
    At the end of set i, whoever won *more points until that point* is predicted
    to be the winner.
    
    If it is a tie, whoever had previously been in the lead is marked as the
    prediction.
    '''
    def __init__(self, raw_data, match_to_examine):
        # Initialize using MatchStats to get the match details
        MatchStats.__init__(self, raw_data, match_to_examine)
        return
    
    def fit(self):
        # Calculate cumulative points won by each player
        self.p1_point_cumulative = np.cumsum(self.point_victors.values == 1)
        self.p2_point_cumulative = np.cumsum(self.point_victors.values == 2)
        
        # Identify the end of each set
        self.set_change_points = np.where(np.diff(self.match['set_no']) > 0)[0]
        self.set_change_points = np.append(self.set_change_points, len(self.point_victors) - 1)
        
        return
    
    def prediction(self):
        '''
        Output: Prediction at the end of each set to predict the winner of the match.
        '''
        predictions = []
        previous_pred = None
        
        for i in self.set_change_points:
            # Get cumulative points up to the end of the current set
            p1_points = self.p1_point_cumulative[i]
            p2_points = self.p2_point_cumulative[i]
            
            # Determine the current prediction
            if p2_points > p1_points:
                current_pred = 2
            else: 
                current_pred = 1
            
            predictions.append(current_pred)
            previous_pred = current_pred
        
        # Return the predictions
        return np.array(predictions)


# TODO
#class CumulativeGameWinnerModel:
class CumulativeGameWinnerModel:
    '''
    At the end of set i, whoever won *more games until that point* is predicted
    to be the winner.
    
    If it is a tie, whoever had previously been in the lead is marked as the
    prediction.
    '''
    def __init__(self, raw_data, match_to_examine):
        # Initialize using MatchStats to get the match details
        MatchStats.__init__(self, raw_data, match_to_examine)
        return
    
    def fit(self):
        # Calculate cumulative games won by each player
        self.p1_game_cumulative = np.cumsum(self.game_victors == 1)
        self.p2_game_cumulative = np.cumsum(self.game_victors == 2)
    
        # Identify the end of each set (your existing logic)
        self.set_change_points = np.where(np.diff(self.match['set_no']) > 0)[0]
        self.set_change_points = np.append(self.set_change_points, len(self.game_victors) - 1)
        
        return
    
    def prediction(self):
        '''
        Output: Prediction at the end of each set to predict the winner of the match.
        '''
        game_predictions = []
        previous_game__pred = None
        
        for i in self.set_change_points:
            # Get cumulative points up to the end of the current set
            p1_games = self.p1_game_cumulative[i]
            p2_games = self.p2_game_cumulative[i]
            
            # Determine the current prediction
            if p2_games > p1_games:
                current_game_pred = 2
            else: 
                current_game_pred = 1
            
            game_predictions.append(current_game_pred)
            previous_game__pred = current_game_pred
        
        # Return the predictions
        return np.array(game_predictions)

# TODO
#class CumulativeUnfErrModel:
class CumulativeUnfErrModel:
    '''
    At the end of set i, whoever has less unforced errors is predicted
    to be the winner.
    
    If it is a tie, whoever had previously been in the lead is marked as the
    prediction.
    '''
    def __init__(self, raw_data, match_to_examine):
        # Initialize using MatchStats to get the match details
        MatchStats.__init__(self, raw_data, match_to_examine)
        return
    
    def fit(self):
        # Calculate cumulative errors by each player
        self.p1_error_cumulative = np.cumsum(self.unf_err == 1)
        self.p2_error_cumulative = np.cumsum(self.unf_err == 2)
    
        # Identify the end of each set (your existing logic)
        self.set_change_points = np.where(np.diff(self.match['set_no']) > 0)[0]
        self.set_change_points = np.append(self.set_change_points, len(self.unf_err) - 1)
        
        return
    
    def prediction(self):
        '''
        Output: Prediction at the end of each set to predict the winner of the match.
        '''
        error_predictions = []
        previous_error__pred = None
        
        for i in self.set_change_points:
            # Get cumulative points up to the end of the current set
            p1_errors = self.p1_error_cumulative[i]
            p2_errors = self.p2_error_cumulative[i]
            
            # Determine the current prediction
            if p2_errors < p1_errors:
                current_error_pred = 2
            else: 
                current_error_pred = 1
            
            error_predictions.append(current_error_pred)
            previous_error__pred = current_error_pred
        
        # Return the predictions
        return np.array(error_predictions)



###################
# Demo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__=="__main__":
    import tennis_data
    import dynamic_model1 as dm1

    
    my_match = matches[1]
    stats = MatchStats(df_raw, my_match)
    
    
    print("Actual:")
    print(np.repeat( stats.winner_id, 5 ))

    print("Per-set prediction based on set winners:")
    model1 = SetWinnerModel(df_raw, my_match)
    print( model1.prediction() )
    
    print("Based on current set leader:")
    model2 = CumulativeSetWinnerModel(df_raw, my_match)
    model2.fit()
    print( model2.prediction() )

    print("Based on current point leader:")
    model3 = CumulativePointWinnerModel(df_raw, my_match)
    model3.fit()
    print( model3.prediction() )

    print("Based on current game leader:")
    model4 = CumulativeGameWinnerModel(df_raw, my_match)
    model4.fit()
    print( model4.prediction() )

    print("Based on current error leader:")
    model5 = CumulativeUnfErrModel(df_raw, my_match)
    model5.fit()
    print( model5.prediction() )
    
    print("Based on our model:")
    model6 = dm1.DynamicTennisModel(df_raw, my_match)
    model6.fit()
    print( model6.prediction() )
    
    df_raw = tennis_data.load_2023()
    matches = df_raw['match_id'].unique()

    # Initialize a dictionary to store counts for each model
    model_names = {
        "SetWinnerModel": SetWinnerModel,
        "CumulativeSetWinnerModel": CumulativeSetWinnerModel,
        "CumulativePointWinnerModel": CumulativePointWinnerModel,
        "CumulativeGameWinnerModel": CumulativeGameWinnerModel,
        "CumulativeUnfErrModel": CumulativeUnfErrModel,
        "DynamicTennisModel": dm1.DynamicTennisModel,  # Include DynamicTennisModel
    }
    
    # Initialize count arrays and reach counters
    count_correct = {model_name: np.zeros(5) for model_name in model_names}
    reach_count = {model_name: np.zeros(5) for model_name in model_names}
    total_matches = 0

    for match_id in matches:
        stats = MatchStats(df_raw, match_id)
        total_matches += 1
        
        for model_name, model_class in model_names.items():
            model_instance = model_class(df_raw, match_id)
            model_instance.fit()
            predictions = model_instance.prediction()
            
            # Check if predictions are valid (not NaN)
            valid_predictions = ~np.isnan(predictions)

            # Compare predictions to actual match winner
            correct_predictions = (predictions[valid_predictions] == stats.winner_id).astype(int)
            
            # Get the valid length
            valid_length = np.sum(valid_predictions)
            
            # Update count_correct and reach_count for valid predictions
            count_correct[model_name][:valid_length] += correct_predictions
            reach_count[model_name][:valid_length] += 1  # Count valid predictions

    # Calculate percentages for each model
    percentage_correct = {}
    for model_name, counts in count_correct.items():
        percentages = np.zeros(5)
        for i in range(5):
            if reach_count[model_name][i] > 0:
                if i >= 3:  # Use reach count for 4th set
                    percentages[i] = (counts[i] / reach_count[model_name][i]) * 100
                else:  # For the first three sets, use total matches
                    percentages[i] = (counts[i] / total_matches) * 100
        percentage_correct[model_name] = percentages

    # Print the results
    for model_name, percentages in percentage_correct.items():
        print(f"{model_name}:")
        for i in range(4):  # Only for sets 1-4
            print(f"  Set {i + 1}: {percentages[i]:.2f}%")
        print()  # Add a newline for better readability

    # Create a DataFrame for plotting (only using the first 4 sets)
    plot_data = pd.DataFrame({
        'Model': list(percentage_correct.keys()),
        'After Set 1': [percentage_correct[model][0] for model in percentage_correct],
        'After Set 2': [percentage_correct[model][1] for model in percentage_correct],
        'After Set 3': [percentage_correct[model][2] for model in percentage_correct],
        'After Set 4': [percentage_correct[model][3] for model in percentage_correct],
    })

    # Melt the DataFrame to long format for Seaborn
    plot_data_long = plot_data.melt(id_vars='Model', var_name='Set', value_name='Percentage')

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data_long, x='Set', y='Percentage', hue='Model', marker='o')
    plt.title('Model Prediction Accuracy by Set')
    plt.ylim(0, 100)
    plt.xlabel('Set')
    plt.ylabel('Percentage Correct')
    plt.legend(title='Model')
    plt.grid()
    plt.show()

## This also prints out a "prediction" for the end of set 5 (the end of the match). This will be
## 100% accurate for any model relying on # of sets won as the winner of set 5 will also have won
## the match.



