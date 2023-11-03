
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
    'font.size': 20,
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

plt.rcParams.update({'font.size': 20})

# vColors = ["#006e64", "#c17150", "#7cab53", "#344B47", "#96B1AC", "#416391", "#7795C7"]
# vColors = ['#006e64', '#317454', '#547944', '#767a38', '#987838', '#a37b3b', '#ae7d3f', '#ba7f43', '#b08a3d', '#a3963d', '#91a145', '#7cab53']
vColors = ['#006e64', '#734848', '#004e4a', '#c17150', '#c8f0e6', '#ffbb00', '#dcfae9', '#a1d9cc', '#1e8c82']
sns.set_palette(sns.color_palette(vColors))
