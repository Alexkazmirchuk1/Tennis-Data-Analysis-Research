
import pandas as pd

# TODO: only if necessary, make this operating system agnostic.
DATA_FOLDER = 'data/'

# TODO: programmatically filter to this
five_sets_2021 = [
    '2021-wimbledon-1104', 
    '2021-wimbledon-1105', 
    '2021-wimbledon-1107', 
    '2021-wimbledon-1108', 
    '2021-wimbledon-1122', 
    '2021-wimbledon-1123', 
    '2021-wimbledon-1125', 
    '2021-wimbledon-1126', 
    '2021-wimbledon-1127', 
    '2021-wimbledon-1136', 
    '2021-wimbledon-1140', 
    '2021-wimbledon-1142', 
    '2021-wimbledon-1144', 
    '2021-wimbledon-1158', 
    '2021-wimbledon-1163', 
    '2021-wimbledon-1214',
    '2021-wimbledon-1215', 
    '2021-wimbledon-1216', 
    '2021-wimbledon-1223', 
    '2021-wimbledon-1316', 
    '2021-wimbledon-1402', 
    '2021-wimbledon-1403', 
    '2021-wimbledon-1406', 
    '2021-wimbledon-1408', 
    '2021-wimbledon-1502'
]

def load_2021(FILE=DATA_FOLDER+'2021-wimbledon-data.csv', exclusions=['2021-wimbledon-1149']):
    '''
    Outputs a Pandas dataframe, loading the file while excluding 
    the given match_ids.
    
    Inputs: FILE: string; location of 2021 Wimbledon data.
            exclusions: list of strings; match_ids to ignore. 
            Default includes match_id 2021-wimbledon-1149
            
    TODO: write code to ease pulling matches with 5 full sets (currently 
    hard-coded in Main.ipynb as of 29 Jul 2024)
    '''
    df = pd.read_csv(FILE)
    mask = ~ df['match_id'].isin(exclusions)
    df = df[mask]
    
    # helps searching by last name/family names
    # magic string processing syntax.
    df['p1_lastname'] = df['player1'].str.split().str[1:].str.join(' ')
    df['p2_lastname'] = df['player2'].str.split().str[1:].str.join(' ')
    
    return df
    
def load_2023(FILE=DATA_FOLDER+'2023-wimbledon-data.csv', exclusions=[]):
    '''
    Outputs a Pandas dataframe, loading the file while excluding 
    the given match_ids.
    
    Inputs: FILE: string; location of 2023 Wimbledon data.
            exclusions: list of strings; match_ids to ignore. 
            Default is an empty list (include all 2023 data)
            
    TODO: write code to ease pulling matches with 5 full sets (currently 
    hard-coded in Main.ipynb as of 29 Jul 2024)
    '''
    df = pd.read_csv(FILE)
    mask = ~ df['match_id'].isin(exclusions)
    df = df[mask]
    
    # helps searching by last name/family names
    # magic string processing syntax.
    df['p1_lastname'] = df['player1'].str.split().str[1:].str.join(' ')
    df['p2_lastname'] = df['player2'].str.split().str[1:].str.join(' ')
    
    return df

