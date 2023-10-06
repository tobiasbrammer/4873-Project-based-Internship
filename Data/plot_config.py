
# Configuration file for setting plot styles

import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.constrained_layout.use': True,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',
    'font.size': 10,
    'axes.edgecolor': 'grey',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.labelcolor': 'dimgrey',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.axisbelow': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})

vColors = ["#006e64", "#c17150", "#7cab53", "#bc4878", "#7e60b7", "#ffbb00"]
sns.set_palette(sns.color_palette(vColors))
