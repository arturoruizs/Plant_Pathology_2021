import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def display_grid(samples, titles, rows, cols,figsize=(12, 6)):
    """Displays examples in a grid."""
    
    #Aqui en lugar de un array con los arrays de imagenes se pasan las rutas de las imagenes
    assert len(samples)==(rows*cols), 'Mismatch between df length and input sizes'
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    i = 0
    for r in range(rows):
        for c in range(cols):
            print(samples[i])
            ax[r, c].imshow(plt.imread(f"train_images/{samples[i]}"), cmap='gray')
            ax[r, c].set_title(titles[i])
            ax[r, c].set_xticklabels([])
            ax[r, c].set_yticklabels([])
            i += 1
    fig.tight_layout()
    plt.show()

def one_hot_labels(df):

    df['labels'] = df['labels'].str.split(' ')
    mlb = MultiLabelBinarizer()

    one_hot_array = mlb.fit_transform(df['labels'])
    one_hot_series = pd.Series(one_hot_array.tolist()) 


    return one_hot_series

