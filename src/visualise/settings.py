def apply_mpl_settings():
    import matplotlib as mpl
    mpl.rcParams['axes.grid.which'] = 'minor'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['figure.figsize'] = (10, 12)
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True


def apply_grid(ax):
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax.grid(which='major', linestyle='-', linewidth='0.7', color='black', alpha=0.7)


def apply_lim(ax):
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)


def apply_defaults(ax):
    apply_grid(ax)
    apply_lim(ax)