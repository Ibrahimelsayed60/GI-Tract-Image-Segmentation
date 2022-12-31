from matplotlib.colorbar import gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as  plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl
import  datagen



def plot_classes_density(df):
    plt.figure(figsize=(10,10))
    bar = plt.bar([1,2,3], 100*np.mean(df.iloc[:, 1:4] != "",axis = 0))
    plt.title("Percent Training Images with mask", fontsize=16)
    plt.ylabel("Percent of Train Images")
    plt.xlabel("class Types")

    labels = ['large_bowel', 'small_bowel', 'stomach']

    for rect,  lbl in zip(bar,labels):
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width() / 3 , height, lbl, ha = "center", va = "bottom", fontsize=12)

    plt.ylim((0,50))
    plt.show()


def plot_mask(df,  colors, labels, batch_size):
    
    list_indices_of_mask_random = list(df[df['large_bowel'] != ""].sample(batch_size).index())
    list_indices_of_mask_random += list(df[df['small_bowel'] != ""].sample(batch_size * 2).index())
    list_indices_of_mask_random += list(df[df['stomach'] != ""].sample(batch_size * 3).index())

    batches_from_datagen = datagen.DataGenerator(df[df.index.isin(list_indices_of_mask_random)],shuffle=True)

    num_rows = 6

    fig = plt.figure(figsize=(10,15))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=2)
    patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]
    cmap1 = mpl.colors.ListColormap(colors[0])
    cmap2 = mpl.colors.ListColormap(colors[1])
    cmap3 = mpl.colors.ListColormap(colors[2])

    for i in range(num_rows):
      images, mask = batches_from_datagen[i]
      sample_img = images[0, :, :, 0]
      mask1 = mask[0, :, :,0]
      mask2 = mask[0, :, :,1]
      mask3 = mask[0, :, :,2]

      ax0 = fig.add_subplot(gs[i,0])
      im = ax0.imshow(sample_img, cmap="bone")

      ax1 = fig.add_subplot(gs[i,1])


      if i==0: 
        ax0.set_title("Image", fontsize=15, weight="bold", y = 1.02)
        ax1.set_title("Mask", fontsize=15, weight="bold", y=1.02)
        plt.legend(handles=patches, bbox_to_anchor = (1.1, 0.65), loc=2, borderaxespad = 0.4, fontsize=14, title="Mask Labels", title_fontsize=14, edgecolor="black", facecolor="#c5c6c7" )
      
      l0 = ax1.imshow(sample_img, cmap="bone")
      l1 = ax1.imshow(np.ma.masked_where(mask1 == False, mask1), cmap=cmap1, alpha = 1)
      l2 = ax1.imshow(np.ma.masked_where(mask2 == False, mask2), cmap=cmap2,  alpha = 1)
      l3 = ax1.imshow(np.ma.masked_where(mask3 == False, mask3), cmap = cmap3, alpha = 1)

      _ = [ax.set_axis_off() for ax in [ax0, ax1]]
      colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3]]


    





