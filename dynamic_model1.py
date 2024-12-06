import numpy as np
from matplotlib import pyplot as plt
from alt_models import MatchStats

#################################

# Functions

# Probability Functions

# Find the probability of winning a game given probability winning a point
def prob_hold(p):
    """
    Probability the serving player holds.
    
    Input: p, float, success probability; in interval [0,1].
    Output: probability, float, in interval [0,1].
    """
    q = 1-p
    output = p**4 + 4*(p**4)*q + (20*(p**3)*(q**3)) * ((p**2)/(1 - 2*p*q))
    return output

# Find the probability of winning a set given probability winning a game
def prob_win_set(p):
    """
    Probability serving player holds.
    
    Input: p, float, success probability; in interval [0,1].
    Output: probability, float, in interval [0,1].
    """
    q = 1-p
    output = p**6 + 6*(p**6)*q + 21*(p**6)*(q**2) + 56*(p**6)*(q**3) + 126*(p**6)*(q**4) + 42*(p**7)*(q**5) + 924*(p**7)*(q**6)
    return output

# Find the probability of winning a match given probability winning a set
def prob_win_match(pm):
    """
    Probability serving player wins match.
    
    Input: p, float, success probability; in interval [0,1].
    Output: probability, float, in interval [0,1].
    """
    qm = 1-pm
    output = pm**3 + 3*(pm**3)*qm + 6*(pm**3)*(qm**2)
    return output

def get_serve_probability(match_data, player):
    """
    Inputs:
        match_data : pandas DataFrame
        player : integer index of player (0, 1, 2 for 2021 for example). 
            Seems to be an index relative to the match being played.
    
    Outputs:
        p_array : 
    """
    
    server_no = match_data['server'].values
    point_victor = match_data['point_victor'].values

    # Purpose: build a list-like of a cumulative rate of winning the point when 
    # the player serves. For example, if the player has served twice and won 
    # one of those points, the value would be 1/2=0.5. If on the following 
    # point the player serves again, and wins the point again, the value gets 
    # updated to (1+1)/(2+1) = 0.66. If on the following point they serve and 
    # lose, the following value is (2+0)/(3+1) = 0.5.
    #
    # The value is carried forward on any points the player is not serving on.
    
    #serve_point_won = 0
    #num_serves = 0
    #p_array = []
    #
    #for index in range(len(server_no)):
    #    if player == server_no[index]:
    #        num_serves += 1
    #
    #        if player == point_victor[index]:
    #            serve_point_won += 1
    #
    #    if num_serves == 0:
    #        p_array.append(0)
    #    else:
    #        p_array.append(serve_point_won / num_serves)
    #
    #p_array = np.array(p_array)
    
    player_is_server = (server_no == player)
    player_served_and_won = np.logical_and(player_is_server, point_victor == player)

    initial_serves = 50
    initial_wins = 25

# Compute cumulative values from the match data
    num_serves_cumulative = initial_serves + np.cumsum(player_is_server)
    won_point_cumulative = initial_wins + np.cumsum(player_served_and_won)

    # if player hasn't served, a calculation 0/0 would occur.
    # impute the probability below with 0/1 instead.
    num_serves_cumulative[num_serves_cumulative==0] = 1
    
    p_array = won_point_cumulative/num_serves_cumulative
    
    
    return p_array


def prob_win_independent_game(p1, p2):
    """
    .....
    
    Inputs:
        p1 : 
        p2 : 
        
    Outputs:
        ps1 : 
        ps2 : 
    """
    p1_holds_game = prob_hold(p1)
    p1_concedes_game = (1 - p1_holds_game)

    p2_holds_game = prob_hold(p2)
    p2_concedes_game = (1 - p2_holds_game)

    ps1 = (p1_holds_game + p2_concedes_game) / 2 # Prob p1 wins any independent game
    ps2 = (p2_holds_game + p1_concedes_game) / 2 # Prob p2 wins any independent game

    return ps1, ps2

# Momentum Modifier Functions

# change probability if player wins set
def modify_momentum(match_data, probability_array, player, r=1.9, q=0.4):
    '''
    Adjust the probability array to reflect momentum changes based on match outcomes.

    Inputs:
        match_data : pandas DataFrame for the match
        probability_array : array of probabilities to modify
        player : player ID (1 or 2)
        r : momentum boost factor (default=1.3)
        q : penalty factor for losses (default=0.4)
    
    Outputs:
        probability_array : modified probability array
    '''
    
    set_victor_array = match_data['set_victor'].values
    won_set_array = set_victor_array == player
    lost_set_array = set_victor_array == (3 - player)
    
    # Cumulative sets won and lost
    n = np.cumsum(won_set_array)
    lost = np.cumsum(lost_set_array)
    
    # Stabilize large values
    loss_factor = 1 + (q * lost)
    
    # Adjust probabilities
    prob_boost = ((np.exp(n * np.log(r)) - 1) / np.exp(n * np.log(r)))
    probability_array = (prob_boost + (1 - prob_boost) * probability_array) ** loss_factor
    
    # Clip to ensure probabilities stay in [0, 1]
    probability_array = np.clip(probability_array, 0, 1)
    
    return probability_array

def modify_momentum_err(match_data, momentum_array, player, s=0.0035):
    '''
    Adjusts a player's momentum based on the cumulative unforced errors 
    made by them and their opponent.

    For each additional unforced error:
        - The player's momentum decreases by `s` (default 0.0035).
        - The player's momentum increases by `s` for each error their opponent commits.

    Inputs:
        match_data : pandas DataFrame
            DataFrame containing match statistics, including unforced error columns.
        momentum_array : numpy array
            Array of current momentum values for the player.
        player : int
            Player index (1 or 2).
        s : float, optional
            Scaling factor for momentum adjustments, default is 0.0035.

    Outputs:
        momentum_array : numpy array
            Updated momentum array after considering unforced errors.
    '''

    # Extract unforced error data for the player and opponent
    player_errors = match_data[f'p{player}_unf_err'].values[:len(momentum_array)]
    opponent_errors = match_data[f'p{3-player}_unf_err'].values[:len(momentum_array)]

    # Calculate cumulative unforced errors for player and opponent
    cumulative_player_errors = np.cumsum(player_errors == 1)
    cumulative_opponent_errors = np.cumsum(opponent_errors == 1)

    # Advantage based on the difference in unforced errors
    error_advantage = cumulative_opponent_errors - cumulative_player_errors

    # Update momentum, ensuring it stays non-negative
    momentum_array = np.maximum(momentum_array + s * error_advantage, 0)

    return momentum_array

def points_scored(match_data, points_array, player, uu=1.005, cc=0.0001):
    '''
    Adjusts a player's "points" based on the cumulative points won and lost during a match.

    When a player wins a point:
        - Their points increase by a factor determined by the parameter `uu`.
    When they lose a point:
        - Their points decrease by a factor scaled by the parameter `cc`.

    Inputs:
        match_data : pandas DataFrame
            DataFrame containing match statistics, including point outcomes.
        points_array : numpy array
            Array of current points for the player at each stage of the match.
        player : int
            Player index (1 or 2).
        uu : float, optional
            Growth factor for winning points, default is 1.005.
        cc : float, optional
            Decay factor for losing points, default is 0.0001.

    Outputs:
        points_array : numpy array
            Updated points array after considering points won and lost.
    '''
    
    # Calculate cumulative points won and lost
    points_victor_array = match_data['point_victor'].values
    won_points = points_victor_array == player
    lost_points = points_victor_array == (3 - player)
    
    # Total points won by the player
    ww = np.sum(won_points)
    
    # Cumulative lost points for scaling
    mm = np.cumsum(lost_points)
    
    # Calculate prefactor for scaling points won
    prefactor = (((uu**ww) - 1) / (uu**ww) + (1 / (uu**ww)))  # A scalar value
    
    # Update points array
    points_array = prefactor * points_array - mm * cc
    
    # Ensure points do not drop below zero
    points_array = np.maximum(points_array, 0)
    
    return points_array
    
###########################
# The main class for these simulations/scores consists of the following general steps: 
# 
# 1 - Calculate serve probability (points_on_serves / total serves)
# 2 - Calculate probability of winning a game from serve probability for each player
# 3 - Calculate probability of winning a set from the player win game probability
# 4 - Calculate probability of winning a match from the player win set probability
# 5 - Graph the probability of winning the set as point values increase
import numpy as np

class DynamicTennisModel:
    def __init__(self, raw_data, match_to_examine, rv=1.25, sv=0.005, qv =0.4, uuv = 1.005, ccv =0.0001,
        palette=plt.cm.Set1):
        
        # Collect basic information about the match
        MatchStats.__init__(self, raw_data, match_to_examine)
        
        ###
        #self.max_length = 0
        #self.p1_momentum = []
        #self.p2_momentum = []

        self.sv = sv
        self.rv = rv
        self.qv = qv
        self.uuv = uuv
        self.ccv = ccv
        
        # colormap to use to generate colors associated with the players.
        self.palette = palette
        
        return

    # 1 - Get serve probabilities
    def get_serve_probabilities(self, debug=False):
        '''
        Output probabilities of each player winning a serve, given the current 
        internal state of the system.
        
        Inputs: 
            debug : boolean; whether to print diagnostic text; default False
        
        Outputs:
            p1_probability,
            p2_probability : floats; probabilities of this event.
        '''
        p1_probability = get_serve_probability(self.match, 1)
        p2_probability = get_serve_probability(self.match, 2)
    
        if debug:
            print("Probability of winning a serve")
            print(p1_probability)
            print(p2_probability)
    
        # Get the maximum length between both probability arrays
        self.max_length = max(len(p1_probability), len(p2_probability))

        return p1_probability, p2_probability
    

    #  2 - Get probability of winning the game
    def get_game_probabilities(self, p1_probability, p2_probability, debug=False):
        '''
        ..............
        
        Inputs:
            p1_probability : 
            p2_probability : 
            debug : boolean; whether to print diagnostic text; default False
            
        Outputs:
            pg1_array, pg2_array : arrays; 
        '''
        #pg1_array = []
        #pg2_array = []
        #
        #for index in range(self.max_length):
        #    pg1, pg2 = prob_win_independent_game(p1_probability[index], p2_probability[index])
        #
        #    pg1_array.append(pg1)
        #    pg2_array.append(pg2)

        pg1_array, pg2_array = prob_win_independent_game(p1_probability[:self.max_length], p2_probability[:self.max_length])

        if debug:
            print("Probability of winning the game")
            print(pg1_array)
            print(pg2_array)

        return pg1_array, pg2_array
    
    # 3 - Get probability of winning the set.
    def get_set_probabilities(self, pg1_array, pg2_array, debug=False):
        '''
        ............
        
        Inputs:
            pg1_array : 
            pg2_array : 
            debug : boolean; whether to print diagnostic text; default False
            
        Outputs:
            ps1_array, ps2_array : arrays; ......
        '''
        #ps1_array = []
        #ps2_array = []
        #
        #for index in range(self.max_length):
        #    ps1_array.append(prob_win_set(pg1_array[index]))
        #    ps2_array.append(prob_win_set(pg2_array[index]))

        ps1_array = prob_win_set(pg1_array[:self.max_length])
        ps2_array = prob_win_set(pg2_array[:self.max_length])

        if debug:
            print("Probability of winning the set")
            print(ps1_array)
            print(ps2_array)


        return ps1_array, ps2_array
    
    # 4 - Get probability of winning the match
    def get_match_probabilities(self, ps1_array, ps2_array, debug=False):
        '''
        ............
        
        Inputs:
            ps1_array : 
            ps2_array : 
            debug : boolean; whether to print diagnostic text; default False
            
        Outputs:
            pm1_array, pm2_array : arrays; ......
        '''
        #pm1_array = []
        #pm2_array = []

        #pm1_array = np.zeros(self.max_length)
        #pm2_array = np.zeros(self.max_length)
        # 
        #for index in range(self.max_length):
        #    #pm1_array.append(prob_win_match(ps1_array[index]))
        #    #pm2_array.append(prob_win_match(ps2_array[index]))
        #    pm1_array[index] = prob_win_match(ps1_array[index])
        #    pm2_array[index] = prob_win_match(ps2_array[index])
        
        # Since the above calculation is a non-recursive polynomial calculation, 
        # calculate them with array arithmetic.
        pm1_array = prob_win_match(ps1_array[:self.max_length])
        pm2_array = prob_win_match(ps2_array[:self.max_length])

        if debug:
            print("Probability of winning the match")
            print(pm1_array)
            print(pm2_array)

        return pm1_array, pm2_array
    
    def update_momentum(self, pm1_array, pm2_array, debug=False):
        '''
        ........
        
        Inputs:
            pm1_array, pm2_array : 
            debug : boolean; whether to print diagnostic text; default False
            
        Outputs:
            p1_momentum11, p2_momentum22 : .......
        '''
        
        p1_momentum = modify_momentum(self.match,  pm1_array, 1, r=self.rv, q=self.qv)
        p2_momentum = modify_momentum(self.match, pm2_array, 2, r=self.rv, q=self.qv)

        p1_momentum1 = modify_momentum_err(self.match, p1_momentum, 1, s=self.sv)
        p2_momentum2 = modify_momentum_err(self.match, p2_momentum, 2, s=self.sv)

        p1_momentum11 = points_scored(self.match, p1_momentum1, 1, uu=self.uuv, cc=self.ccv)
        p2_momentum22 = points_scored(self.match, p2_momentum2, 2, uu=self.uuv, cc=self.ccv)

        if debug:
            print(p1_momentum1)
            print(p2_momentum2)

        return p1_momentum11, p2_momentum22
    
    # 5 - Graph
    def add_graph_decorations(self, ax):
        '''
        Given a pyplot Axis handle, put decorations such as markers for the set
        on it. Uses data from the match saved in the instantiation of this object.
        
        Inputs: ax, axis handle.
        Outputs: None (Artists are added to the same Axis)
        '''
        from matplotlib import ticker
        
        set_change_points = list([0, *self.set_change_points])
        game_change_points = np.where(np.diff(self.match['game_no'])>0)[0]
        
        ax.xaxis.set_major_locator(ticker.FixedLocator(set_change_points))
        ax.tick_params(which='major',length=8)
        ax.xaxis.grid(True, which='major', linestyle='--', color='#333')
        
        ax.set_xticklabels([f"Set {_i+1}" for _i in range(len(set_change_points))])
        #ax.tick_params(axis="x", ha="left")
        plt.setp(ax.get_xticklabels(), horizontalalignment='left')
        
        ax.xaxis.set_minor_locator(ticker.FixedLocator(game_change_points))
        ax.tick_params(which='minor',length=4)
        ax.xaxis.grid(False, which='minor')
        
        #for index, _x in enumerate(set_change_points):
        #    #ax.axvline(x=_x, color='#333', linestyle='--', zorder=-1000)
        #    ax.text(_x, ax.get_ylim()[0], f"Set {index+1}", 
        #    va='bottom', ha='left', rotation=90, 
        #    bbox={'facecolor':'#fff', 'edgecolor':'#333', 'boxstyle':'square,pad=0.2'})
        
        
        for _k in range(2):
            for spine in ['bottom', 'top', 'right']:
                ax.spines[spine].set_visible(False)
        ax.yaxis.grid(True)
        
        return None
    
    def graph_momentum(self, ax=None):
        '''
        Creates a plot of the dynamically evolving momentum values for the match 
        studied. It is assumed self.p1_momentum and self.p2_momentum have been 
        created and plotted.
        
        Inputs: None
        Outputs: fig,ax : pyplot figure/axis pair associated with the plot created.
        '''
        from matplotlib import pyplot as plt
        
        if ax is None:
            fig,ax = plt.subplots()
        else:
            fig = ax.get_figure()
        
        self.add_graph_decorations(ax)
        
        ax.plot(range(len(self.p1_momentum)), self.p1_momentum, color=self.palette(0), label=f"{self.player1_surname}")
        ax.plot(range(len(self.p2_momentum)), self.p2_momentum, color=self.palette(1), label=f"{self.player2_surname}")
        
        ax.legend(loc='upper right')
        
        return fig,ax

    def prediction(self, verbose=False):
        '''
        Inputs: verbose : boolean; whether to print diagnostic text; default False
        
        Outputs: 
            result_array : array; entry i indicates 
            whether the current state of the momentum model at the end of set i 
            correctly predicts the final result of the match; as an integer 
            (0 = incorrectly predicted, 1 = correctly predicted).
        '''
        set_change_values = [
            self.p1_momentum[self.set_change_points], 
            self.p2_momentum[self.set_change_points]
        ]
        
        # pad result_array with NaN in matches with fewer than five sets.
        pred = np.nan * np.zeros(5)
        
        # decision: performance_diff>0?
        performance_diff = set_change_values[1] - set_change_values[0]
        pred[:len(performance_diff)] = 1 + (performance_diff>0).astype(int)

        return pred
    #
    
    def evaluate_prediction(self, pred):
        return (pred == self.match_winner).astype(int)
    
    def fit(self, debug=False):
        '''
        Runs the model for the match this object was instantiated with.
        Updates self.p1_momentum and self.p2_momentum.
        
        Inputs : debug; boolean, whether to print diagnostic text; default False.
            This flag is automatically passed forward to intermediate functions.
        Outputs: None
        '''
        # possible optimizable?
        #self.p1_momentum, self.p2_momentum = self.update_momentum(
        #    *self.get_match_probabilities( # 4
        #        *self.get_set_probabilities( # 3
        #            *self.get_game_probabilities( # 2
        #                *self.get_serve_probabilities(debug), debug), debug), debug), debug) # 1

        a = self.get_serve_probabilities(debug)
        b = self.get_game_probabilities(*a, debug)
        c = self.get_set_probabilities(*b, debug)
        d = self.get_match_probabilities(*c, debug)
        self.p1_momentum, self.p2_momentum = self.update_momentum(*d, debug)
        
        
        self.p1_momentum = np.array(self.p1_momentum)
        self.p2_momentum = np.array(self.p2_momentum)
        return
    
#########################################

if __name__=="__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import itertools # for shortening double-loop code.

    plt.style.use('ggplot')

    import tennis_data
    
    raw_data = tennis_data.load_2023()
    MATCHES_TO_EXAMINE = raw_data['match_id'].unique()

    # MATCHES_TO_EXAMINE = tennis_data.five_sets_2021
    
    # For 2023: 
    # raw_data = tennis_data.load_2023()

    num = 5

    rvalues = np.linspace(1, 1.8, num)
    svalues = np.linspace(0, 0.006, num)
    qvalues = np.linspace(0.01, 1, num)
    uuvalues = np.linspace(1, 1.005, num)
    ccvalues = np.linspace(0., 0.001, num)

    results = np.zeros( (num, num) )

    # CODE TO BE AUTOMATED/TESTED OVER PARAM VALUES.
    #if True: 
    for i,j in itertools.product( range(num), range(num) ):
        #set1_correct = set1_total = set2_correct = set2_total = set3_correct = set3_total = set4_correct = set4_total = 0
        
        correct_preds = np.zeros((4,num,num), dtype=int)
        num_preds = np.zeros((4,num,num), dtype=int)
        
        # 
        
        print(i,j)
        
        for MATCH_TO_EXAMINE in MATCHES_TO_EXAMINE:
            model = DynamicTennisModel(raw_data, MATCH_TO_EXAMINE,  uuv=uuvalues[j]  , ccv=ccvalues[i])
            model.fit()
            
            pred = model.prediction()
            result_array = model.evaluate_prediction(pred)
            
            # accumulate statistics based on the set number being predicted.
            for k in range(4):
                if not np.isnan(result_array[k]):
                    correct_preds[k,i,j] += result_array[k]
                    num_preds[k,i,j] += 1
        #
        
        # set 4 results
        results[i,j] = correct_preds[3,i,j]/num_preds[3,i,j] # store prediction rate.
        
    if False:
        print(rvalues[i], svalues[j])
        print(f"Predicted winner at set 2 correctly {set1_correct} / {set1_total} times")
        print(f"Predicted winner at set 3 correctly {set2_correct} / {set2_total} times")
        print(f"Predicted winner at set 4 correctly {set3_correct} / {set3_total} times")
        print(f"Predicted winner at set 5 correctly {set4_correct} / {set4_total} times")
        
        results[i,j] = set4_correct/set4_total # store prediction rate
        
    ##########################################

    # visualize results of parameter sweep.
    fig,ax = plt.subplots()
    cax = ax.pcolor(ccvalues, uuvalues, results.T)
    ax.set(xlabel='cc', ylabel='uu')
    fig.colorbar(cax)

    ##

    print(results.T)


