import dynamic_model1 as dm
import tennis_data
from matplotlib import pyplot as plt
import numpy as np

df_raw = tennis_data.load_2021()
matches = df_raw['match_id'].unique()

my_match = matches[3] # TODO: which one corresponds to the Djokovic/Alcaraz match?

# visualize momentum only to help validate upstream changes.

model = dm.DynamicTennisModel(df_raw, my_match)
model.train()

pred = model.prediction()

fig,ax = plt.subplots(2,1, figsize=(6,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})

fig,ax[0] = model.graph_momentum(ax=ax[0])

# plot unforced error diff, from the perspective of player1 (positive numbers better)
model.add_graph_decorations(ax[1])
unf_diff = np.cumsum( model.match['p2_unf_err'].values - model.match['p1_unf_err'].values )

ax[1].plot(np.arange(len(unf_diff)), unf_diff, c='k')
ax[1].set(yticks=[0])
ax[1].yaxis.grid(True)

ax[0].set(ylabel="Performance")
ax[1].set(ylabel="Unf err diff")

fig.show()

