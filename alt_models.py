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
        self.unf_err = self.match['p1_unf_err'][self.match['p1_unf_err'] != 0]
        
        
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




###################
# Demo
if __name__=="__main__":
    import tennis_data
    import dynamic_model1 as dm1
    
    df_raw = tennis_data.load_2023()
    matches = df_raw['match_id'].unique()

    my_match = matches[11]
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
    
    print("Based on our model:")
    model5 = dm1.DynamicTennisModel(df_raw, my_match)
    model5.fit()
    print( model5.prediction() )
    
