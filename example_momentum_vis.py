import dynamic_model1 as dm
import tennis_data
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

plt.rcParams.update({'font.family': 'Monospace'})

# A few matches to try out:
# 2021:
# Nishioka/Bedene: 79
# Kudla/Fokina: 3 (has 5 sets)
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

fig,ax = plt.subplots(2,1, figsize=(8,6), sharex=True, gridspec_kw={'height_ratios':[3,1]})

fig,ax[0] = model.graph_momentum(ax=ax[0])

#######################################
# Also try to plot unforced error diff, from the perspective of player1 (positive numbers better)
unf_diff = np.cumsum( model.match['p2_unf_err'].values - model.match['p1_unf_err'].values )

ax[1].plot(np.arange(len(unf_diff)), unf_diff, c='k')

# fill the plot with a color showing which player currently has fewer total 
# unforced errors to that point.
p1_lead = unf_diff>=0
ax[1].fill_between(np.arange(len(unf_diff)), unf_diff, 0, where = p1_lead, color=plt.cm.Set1(0), alpha=0.5)
ax[1].fill_between(np.arange(len(unf_diff)), 0, unf_diff, where = ~p1_lead, color=plt.cm.Set1(1), alpha=0.5)

ax[1].set(yticks=[0])

model.add_graph_decorations(ax[1])

ax[0].set(ylabel="Performance", xlim=[0,model.match.shape[0]], ylim=[0,1.1])
ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))

ax[1].set(ylabel="Unf err diff")

fig.align_ylabels(ax)

fig.show()

