import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

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
      

def plot_images_from_folder(
    folder_path, rows=1, cols=1, img_format='.png', figsize=(10, 10),
    title_size=10, cmap='binary', cmap_reverse=False, fname=None,
    save_dpi=200, show_plot=True):
    """
    Plot images from a folder in a single figure.

    Args:
        folder_path (str): Path to the folder containing images.
        rows (int): Number of rows in the subplot grid.
        cols (int): Number of columns in the subplot grid.
        figsize (tuple): Figure size (width, height) in inches.
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # === get the images ===
    # image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = [f for f in os.listdir(folder_path) if f.endswith((img_format))]
    image_files.sort()

    # === plot each images ===
    for i, ax in enumerate(axes):
        if i < len(image_files):
            img_path = os.path.join(folder_path, image_files[i])
            img = mpimg.imread(img_path)
            if cmap_reverse==False:
                ax.imshow(img, cmap=plt.colormaps.get_cmap(cmap))
            elif cmap_reverse==True:
                ax.imshow(img, cmap=plt.colormaps.get_cmap(cmap).reversed())
            ax.set_title(image_files[i], fontsize=title_size)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()

    # === show condition ===
    if show_plot==True:
        plt.show()
        # save condition
        if fname!=None:
            plt.savefig(fname, dpi=save_dpi)
        plt.close()
        plt.clf() 
    elif show_plot==False:
        # save condition
        if fname!=None:
            plt.savefig(fname, dpi=save_dpi)
        plt.close()
        plt.clf() 
