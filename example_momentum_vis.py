import dynamic_model1 as dm
import tennis_data


df_raw = tennis_data.load_2021()
matches = df_raw['match_id'].unique()

my_match = matches[15] # TODO: which one corresponds to the Djokovic/Alcaraz match?

# visualize momentum only to help validate upstream changes.

model = dm.MarkovChain(df_raw, my_match)
model.train()

fig,ax = model.graph_momentum()

fig.show()
