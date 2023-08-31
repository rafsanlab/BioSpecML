import matplotlib.pyplot as plt

def plot_rxc(nrows=1, ncols=2, dpi=120, figsize=(9,3), imgs=[], titles=[],
             title_size=10, cmap='binary', show=True, fname=None):
    """
    Plot X number of rows and columns subplot.

    Args:
        nrows: number of rows (1)
        ncols: number of columns
        dpi: dpi value of the figure
        figsize: figure size
        imgs: list of image array to be plotted
        titles: list of titles in string
        title_size: title font size
        cmap: plt's cmap
        show: to show plot or not
        fname: str of image name with img format

    Returns:
        plt.show()
    """  
    _, axs = plt.subplots(nrows, ncols, dpi=dpi, figsize=figsize)
    for i in range(ncols): axs[i].imshow(imgs[i], cmap=cmap)  
    if titles != []:
        for i in range(ncols): axs[i].title.set_text(titles[i])
        for i in range(ncols): axs[i].title.set_size(title_size)
    for i in range(ncols): axs[i].set_axis_off()
    plt.tight_layout()
    if fname != None:
        plt.savefig(fname, transparent=True, dpi=dpi)
    if show==False:
        plt.ioff()
        plt.close()
    elif show==True:
        plt.show(block=False);
        plt.close()
