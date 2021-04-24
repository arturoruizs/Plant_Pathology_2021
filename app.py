import streamlit as st 
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt 
def get_label_frequencies():
    class_freq = train_df['labels'].value_counts()
    labels = class_freq.index.to_list()
    freqs = class_freq.values
    label_df = pd.DataFrame(data = {'labels': labels, 'frequencies': freqs} )
    label_df['labels']= (label_df['labels'].str.replace('_'," ")).str.title()

    st.write('How many times is each class repeated on the training data?')
    fig = px.bar(label_df, x='labels', y='frequencies')
    st.plotly_chart(fig)


def display_grid(xs, titles, rows, cols,figsize=(12, 6)):
    """Displays examples in a grid."""
    
    #Aqui en lugar de un array con los arrays de imagenes se pasan las rutas de las imagenes
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
    st.pyplot(fig)

st.title('Plant Infection Visualization')

#Paths to training images and labels
train_images_path = '/train_images'
train_df_path = 'train.csv'

train_df = pd.read_csv(train_df_path)

get_label_frequencies()

#Plot a Batch of images with different labels
label_df = train_df.groupby('labels').first().reset_index()
labels,images = label_df['labels'],label_df['image']

st.write('These are the different classes of leaves in the dataset')
display_grid(images,labels,3,4)