import numpy as np

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

    num_serves_cumulative = np.cumsum(player_is_server)
    won_point_cumulative = np.cumsum(player_served_and_won)

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
def modify_momentum(match_data, probability_array, player, r=1.3, q=0.4):
    '''
    When a player wins a set their "points" will increase by an exponential amount. 
    When they lose a set their "points" will decrease but this change isn't as significant.
    
    Inputs:
        match_data : pandas DataFrame for the match
        probability_array : array of .....
        
    Outputs:
        probability_array : ...........
    '''
    # n = number of sets won
    #n = 0
    #lost = 0
    #loss_factor = 1
    #x = 1.25
    #
    #for index in range(len(probability_array)):
    #    
    #    won_set = set_victor_array[index] == player
    #
    #    if (won_set):
    #        n += 1
    #        n = n
    #    elif set_victor_array[index] == 3 - player:
    #        lost += 1
    #        # probability_array[index] = probability_array[index]/(2*lost)
    #        loss_factor = 1 + (q * lost)
    #
    #    probability_array[index] = (((r**n)-1)/(r**n) + (1/(r**n))*probability_array[index]) ** loss_factor
    
    set_victor_array = match_data['set_victor'].values
    won_set_array = set_victor_array == player
    lost_set_array = set_victor_array == (3-player)
    
    n = np.cumsum( won_set_array )
    lost = np.cumsum( lost_set_array )
    loss_factor = 1 + (q * lost)
    
    
    probability_array = (((r**n)-1)/(r**n) + (1/(r**n))*probability_array) ** loss_factor

    return probability_array

def modify_momentum_err(match_data, momentum_array, player, s=0.0035):
    '''
    Calculates the total number of unforced errors for a player and their opponent. 
    For each additional unforced error a player commits, they will lose 0.0035 "points" (default).
    For each additional unfrced error their opponent commits, they will gain 0.0035 "points".
    
    Inputs:
        match_data : pandas DataFrame
        momentum_array : array; ......
        player : 
    Outputs:
        momentum_array : ............
    '''

    # unforced error total
    #n = 0
    #m = 0

    # Counting the number of unforced errors.
    
    unf_err_array = match_data[f'p{player}_unf_err'].values
    #for index in range(len(momentum_array)):
    #    unf_err = unf_err_array[index] == 1
    #
    #    if (unf_err):
    #        n += 1

    n = sum(unf_err_array[:len(momentum_array)] == 1)

    unf_err_array_2 = match_data[f'p{3-player}_unf_err'].values
    
    # IMPORTANT: 
    # TODO:
    # Is this what is intended by the model (not the code)? 
    # Currently the code uses n (the total number of unf errors after the fact), 
    # an integer, compared to m, a cumulative number of unf errors for the second 
    # player up to that point, an array.
    #
    # Doesn't make sense because n uses "future" values; and if we're trying to 
    # model a sense of players feeling "overwhelmed" by making more unforced 
    # errors than their opponent, it seems sensible to use data only up to that 
    # point in the match.
    
    #for index in range(len(momentum_array)):
    #    unf_err_2 = unf_err_array_2[index] == 1
    #
    #    if (unf_err_2):
    #        m += 1
    #
    #    if n>= 0:
    #        momentum_array[index] = momentum_array[index]-s*n + s*m
    #        if momentum_array[index] < 0:
    #            momentum_array[index] = 0
    
    # Vectorized version of above.
    m = np.cumsum(unf_err_array_2[:len(momentum_array)] == 1)
    momentum_array = np.maximum(momentum_array + s*(m-n), 0)
    
    
    return momentum_array

def points_scored(match_data, points_array, player, uu= 1.005, cc=0.0001):
    '''
    When a player wins a set their "points" will increase by an exponential amount. 
    When they lose a set their "points" will decrease but this change isn't as significant.
    
    Inputs:
        match_data : pandas DataFrame for the match.
        points_array : array, .....
        player : integer index of player
        uu : parameter (float); default 1.005
        cc : parameter (float); default 0.0001 (or, 10**-4)
    '''
    # w = number of points won
    ww = 0
    mm = 0
    lost_factor=1
    
    points_victor_array = match_data['point_victor'].values
    
    
    # Important:
    # TODO: 
    # Similar to elsewhere, the value of "ww" being used in these calculations 
    # is the sum of point_victor values mathcing that player, for the entire 
    # match; while values for "mm" is dynamically changing for each value 
    # in points_array, during the second loop. Is this what is intended?
    
    won_point = points_victor_array == player
    lost_point = points_victor_array == 3 - player
    
    #for index in range(len(points_array)):
    #    points_victor_array = match_data['point_victor'].values
    #    won_point = points_victor_array[index] == player
    #
    #    if (won_point):
    #        ww += 1
    
    ww = sum(won_point)
    
    #for index in range(len(points_array)):
    #    points_loss_array = match_data['point_victor'].values
    #    lost_point = points_loss_array[index] == 3 - player
    #
    #    if (lost_point):
    #        mm += 1
    #        lost_factor = 1 + (mm * cc)
    #
    #        
    #    points_array[index] = (((uu**ww)-1)/(uu**ww) + (1/(uu**ww)))*points_array[index]-mm*cc
    
    mm = np.cumsum(lost_point)
    prefactor = (((uu**ww)-1)/(uu**ww) + (1/(uu**ww))) # currently a float; not an array (ww is fixed)
    
    points_array = prefactor*points_array-mm*cc
    
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

class MarkovChain:
    def __init__(self, raw_data, match_to_examine, rv=1.25, sv=0.005, qv =0.4, uuv = 1.005, ccv =0.0001 ):
        self.match = raw_data[raw_data['match_id'] == match_to_examine]
        self.player1_name = self.match['player1'].values[0]
        self.player2_name = self.match['player2'].values[0]
        self.max_length = 0
        self.p1_momentum = []
        self.p2_momentum = []

        self.sv = sv
        self.rv = rv
        self.qv = qv
        self.uuv = uuv
        self.ccv = ccv

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
    def graph_momentum(self):
        '''
        Creates a plot of the dynamically evolving momentum values for the match 
        studied. It is assumed self.p1_momentum and self.p2_momentum have been 
        created and plotted.
        
        Inputs: None
        Outputs: fig,ax : pyplot figure/axis pair associated with the plot created.
        '''
        from matplotlib import pyplot as plt
        
        # find where we go from one set number to the next. These are sequential;
        # e.g. self.match['set_no'] could look like [1, 1, 1, 1, 2, 2, 3, 3], 
        # and we would want set_change_points to be [4, 6].
        set_change_points = 1 + np.where( np.diff(self.match['set_no']) > 0 )[0]

        fig,ax = plt.subplots()
        
        ax.plot(range(len(self.p1_momentum)), self.p1_momentum, color="red", label=f"{self.player1_name}")
        ax.plot(range(len(self.p2_momentum)), self.p2_momentum, color="blue", label=f"{self.player2_name}")
        
        
        ax.set(title="Game Flow", xlabel="Point Number", ylabel="Performance Rate")
        ax.legend(loc='upper right')

        ax.text(20, -.04, 'Set 1', verticalalignment='bottom')
        for index, value in enumerate(set_change_points):
            ax.axvline(x=value, color='gray', linestyle='--')
            ax.text(value + 20, -.04, f"Set {index + 2}", verticalalignment='bottom')
        
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
        # Find who was performing better before sets 3 4 and 5
        #set_change_points = []
        #
        #
        #old_entry = 1
        #for index, entry in enumerate(self.match['set_no']):
        #    if entry != old_entry:
        #        set_change_points.append(index + 1)
        #        old_entry = entry
        #        
        set_change_points = 1 + np.where( np.diff(self.match['set_no']) > 0 )[0]
        
        set_victors = self.match['set_victor']
        final_point = set_victors.iloc[-1]
        
        # TODO: replace/reposition code which *prints* per-set predictions.

        self.set_change_points = set_change_points
        
        data = np.vstack([self.p1_momentum[self.set_change_points], self.p2_momentum[self.set_change_points]])
        data = data.T
        
        result_array = self.determine_results(data, final_point)

        result_array = result_array.flatten()

        return result_array
        
    #
    
    def train(self, debug=False):
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
    
    def determine_results(self, data, win, verbose=False):
        '''
        ..........
        
        Inputs:
            data : ...
            win : ...
            verbose : boolean, whether to print diagnostic text; default False
            
        Outputs:
            result_array : integer 0/1 array; whether at the end of set i+1 
            whether the current momentum values correctly predict the end 
            result of the match.
        '''
        predicted_winner = 1+(np.diff(data, axis=1) )

        actual_winner = win*np.ones( np.shape(predicted_winner) )

        winner_number = actual_winner-1
        #winner_number

        # n-by-1 boolean array (True means predicted correctly)
        equal_elements = np.floor(predicted_winner) == winner_number
        
        result_array = equal_elements.astype(int)
        
        return result_array
        
#########################################

if __name__=="__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import itertools # for shortening double-loop code.

    plt.style.use('ggplot')

    import tennis_data
    
    raw_data = tennis_data.load_2021()
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
        set1_correct = set1_total = set2_correct = set2_total = set3_correct = set3_total = set4_correct = set4_total = 0
        
        # 
        
        print(i,j)
        
        for MATCH_TO_EXAMINE in MATCHES_TO_EXAMINE:
            # TODO: redesign code so that files get loaded *once*, outside the loop.
            model = MarkovChain(raw_data, MATCH_TO_EXAMINE,  uuv=uuvalues[j]  , ccv=ccvalues[i])
            model.train()
            #if i == 0 and j == 0:
            #     model.graph_momentum()
            result_array = model.prediction()
        
            try:
                set1_correct += result_array[0,0]
                set1_total += 1
        
                set2_correct += result_array[1,0]
                set2_total += 1
        
                set3_correct += result_array[2,0]
                set3_total += 1
        
                set4_correct += result_array[3,0]
                set4_total += 1
            finally:
                continue
        results[i,j] = set4_correct/set4_total # store prediction rate.
        
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


