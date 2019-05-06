import numpy as np
import matplotlib.pyplot as plt

decisionNode=dict(boxstyle="square",fc="w")
leafNode=dict(boxstyle="circle",fc="w")
arrow_args=dict(arrowstyle="<-", shrinkA = 0,shrinkB=25)


# Define the function in plotting node and leaf
def plotNode(nodeText,centerPt,parentPt,nodeType):
    # Define the property of Node and Leaf

    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                           va='center',ha='center',size=10,bbox=nodeType,arrowprops=arrow_args)

# define the main functions, plotTree
def plotTree(inTree, parentPt):
    numLeafs = inTree.get_num_leafs()   #this determines the x width of this tree
    depth = inTree.get_depth()
    nodeStr='X' + str(inTree.attribute)  + ' < ' + str(inTree.value)  #the text label for this node should be this
    cntrPt = (plotTree.xOff +5/plotTree.totalW, plotTree.yOff)      # location of the node
    plotNode(nodeStr, cntrPt, parentPt, decisionNode)       # plot the node
    plotTree.yOff = plotTree.yOff - 1.5/plotTree.totalD   # Move to next level to change yoff
    if not inTree.left.leaf:  #test to see if the nodes are dictonaires, if not they are leaf nodes
        plotTree(inTree.left,cntrPt)        #recursion
    else:                                             #it's a leaf node print the leaf node
        plotTree.xOff = plotTree.xOff + 5/plotTree.totalW
        leafStr = 'Leaf:' + str(inTree.left.classes)
        plotNode(leafStr, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
    if not inTree.right.leaf:  #test to see if the nodes are dictonaires, if not they are leaf nodes
        plotTree(inTree.right,cntrPt)        #recursion
    else:                                             #it's a leaf node print the leaf node
        plotTree.xOff = plotTree.xOff + 5/plotTree.totalW
        leafStr = 'Leaf:' + str(inTree.right.classes)
        plotNode(leafStr, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
    plotTree.yOff = plotTree.yOff + 1.5/plotTree.totalD

# create the plot of decision tree
def createPlot(inTree):
    fig=plt.figure(1,figsize=(7,7),facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(frameon=False,**axprops)
    plotTree.totalW=float(inTree.get_num_leafs())
    plotTree.totalD=float(inTree.get_depth())
    plotTree.xOff=1/plotTree.totalW
    plotTree.yOff=3
    # Initialize the root location
    init_loc = cntrPt = (plotTree.xOff +5/plotTree.totalW, plotTree.yOff)
    plotTree(inTree, init_loc)
    plt.savefig('treeplot.png', bbox_inches="tight")
