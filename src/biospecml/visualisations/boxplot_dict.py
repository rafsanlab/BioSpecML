"""
Example of dict for reference to be use in plot_df() function, to be 
pass as *plt_args argument when plot_mode is set to 'box'.

Example:

    # dicts for boxplot
    boxprops = dict(facecolor='darkorange', linewidth=1, linestyle='solid', edgecolor='black')
    flierprops = dict(marker='.', markerfacecolor='black', markersize=3, linestyle='None')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    whiskerprops = dict(color='black', linewidth=1) 
    capprops = dict(color='black', linewidth=1)

    # plot a boxplot
    plot_df(df_plot,
            plot_mode='box',
            plt_args={'rot':90, 'patch_artist': True,
                    'flierprops':flierprops,
                    'boxprops':boxprops,
                    'medianprops':medianprops,
                    'whiskerprops':whiskerprops,
                    'capprops':capprops,
                    },
            )

"""

# dicts for boxplot
boxprops = dict(facecolor='darkorange', linewidth=1, linestyle='solid', edgecolor='black')
flierprops = dict(marker='.', markerfacecolor='black', markersize=3, linestyle='None')
medianprops = dict(linestyle='-', linewidth=2, color='black')
whiskerprops = dict(color='black', linewidth=1) 
capprops = dict(color='black', linewidth=1)