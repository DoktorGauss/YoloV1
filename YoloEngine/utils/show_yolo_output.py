
def showImageWithBoundingBoxes(im,bndBx):
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    for outputRect in bndBx:
        oben = outputRect['y'] - outputRect['height']/2
        links = outputRect['x'] - outputRect['width']/2
        obenLinks=(links,oben)
        width = outputRect['width']
        height = outputRect['height']
        # Create a Rectangle patch
        rect = patches.Rectangle(obenLinks,width,height,linewidth=1,edgecolor='r',facecolor='none')
        #if(outputRect['class_name']=='Zahlung' or outputRect['class_name']=='Umsatzsteuer'):
        ax.text(links,oben, outputRect['class_name'] + ' : ' + np.str(outputRect['class_probability']), fontsize=8, color='green')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()