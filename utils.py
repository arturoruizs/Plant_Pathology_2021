def display_grid(xs, titles, rows, cols,figsize=(12, 6)):
    """Displays examples in a grid. Insted of requiring a 2D array, requires the path
       of the images """
    
    assert len(xs.index)==(rows*cols), 'Mismatch between df length and input sizes'
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    i = 0
    for r in range(rows):
        for c in range(cols):
            print(xs[i])
            ax[r, c].imshow(plt.imread(f"train_images/{xs[i]}"), cmap='gray')
            ax[r, c].set_title(titles[i])
            ax[r, c].set_xticklabels([])
            ax[r, c].set_yticklabels([])
            i += 1
    fig.tight_layout()
    plt.show()