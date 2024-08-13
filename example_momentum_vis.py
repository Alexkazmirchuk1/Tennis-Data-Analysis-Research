import dynamic_model1 as dm
import tennis_data
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np


# A few matches to try out:
# 2021:
# Nishioka/Bedene: 79
# 
# 2023:
# Djokovic/Alcaraz: 30
# 

df_raw = tennis_data.load_2023()
matches = df_raw['match_id'].unique()

my_match = matches[30]

# visualize momentum only to help validate upstream changes.

model = dm.DynamicTennisModel(df_raw, my_match)
model.train()

pred = model.prediction()

fig,ax = plt.subplots(2,1, figsize=(6,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})

fig,ax[0] = model.graph_momentum(ax=ax[0])

# plot unforced error diff, from the perspective of player1 (positive numbers better)
unf_diff = np.cumsum( model.match['p2_unf_err'].values - model.match['p1_unf_err'].values )

ax[1].plot(np.arange(len(unf_diff)), unf_diff, c='k')
ax[1].set(yticks=[0])
ax[1].yaxis.grid(True)

model.add_graph_decorations(ax[1])

ax[0].set(ylabel="Performance", xlim=[0,model.match.shape[0]], ylim=[0,1.1])
ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))

ax[1].set(ylabel="Unf err diff")

for _k in range(2):
    for spine in ['bottom', 'top', 'right']:
        ax[_k].spines[spine].set_visible(False)

fig.show()

