from __future__ import print_function
import shapely.geometry as shy
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets import FloatSlider
import numpy as np
import cvxpy as cp
from random import randrange, uniform

import matplotlib.pyplot as plt
import random
import igl
import math
import tripy
import networkx as nx
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.pyplot import figure
colors= np.array(['#85C17E','#E2BC74','#425B8A','#B666D2','#9393b3','#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E' ,'#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E' ,'#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E' ,'#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E' ,'#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E' ,'#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E' ,'#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E' ,'#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E' ,'#85C17E','#E2BC74','#425B8A','#B666D2','#85C17E'   ])
#colors = np.array(['#FFF8DC', '#FFEBCD','#FFE4C4','#FFDEAD','#F5DEB3','#DEB887','#D2B48C','#BC8F8F','#F4A460','#DAA520','#B8860B','#CD853F','#D2691E','#8B4513','#A52A2A','#800000'])
#colors = np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'])
import copy
from numpy import linalg as LA
epsilon=1e-5
from shapely.geometry import Point


def simpleDisplay(figure):
    patches = []
    xmax=0
    ymax=0
    xmin =0
    ymin=0
    k=0
    fig2 = plt.figure(num=None, figsize=(8, 8))
    ax2 = fig2.gca()
    for i in range(len(figure)):
        polygon = Polygon(figure[i], fill=True, facecolor=colors[i],edgecolor='black',label='figure'+str(i))
        ax2.add_artist(polygon)
        if(xmax<np.max(figure[i][:,0])):
            xmax=np.max(figure[i][:,0])
        if(ymax<np.max(figure[i][:,1])):
            ymax=np.max(figure[i][:,1])
        if(xmin>np.min(figure[i][:,0])):
            xmin=np.min(figure[i][:,0])
        if(ymin>np.min(figure[i][:,1])):
            ymin=np.min(figure[i][:,1])
        ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plt.show()

def cross(a,b):
    return a[0]*b[1]-b[0]*a[1]

def is_between(a, b, c): # is point c between point a and b
    if(np.allclose(a,c) or np.allclose(b,c)):
        return False
    cross_product= cross((c-a),(b-a))
    if abs(cross_product) > epsilon:
        return False
    dotproduct = (b-a)@(c-a)
    if dotproduct < 0:
        return False
    squaredlengthba = np.sum(np.square(b-a))
    if dotproduct > squaredlengthba:
        return False
    return True

def isTouching(v1,v2, u1,u2):
    if((np.allclose(v1,u1) and np.allclose(v2,u2)) or  (np.allclose(v2,u1) and np.allclose(v1,u2))):
        return True
    return abs(cross(v1-v2, u1-u2)) < epsilon and (is_between(v1,v2,u1) or is_between(v1,v2,u2) or is_between(u1,u2,v1) or is_between(u1,u2,v2))  
def compute_normal_edge(v1,v2,i,j):
    normal = np.array(v2-v1)
    normal[0], normal[1] = normal[1], -normal[0]
    normal = normal / LA.norm(normal)
    return np.around(np.array(normal), decimals = 1)

def compute_normals_shape(shape):
    normals_of_shape= np.array([compute_normal_edge(shape[i],shape[(i+1)%len(shape)]) for i in range(len(shape))])
    return normals_of_shape

def compute_normals_2_shapes(shape1, shape2): 
    result = np.array([compute_normal_edge(shape1[i],shape1[(i+1)%len(shape1)],shape2[j],shape2[(j+1)%len(shape2)]) for i in range (len(shape1)) for j in range (len(shape2)) if(isTouching(shape1[i],shape1[(i+1)%len(shape1)],shape2[j],shape2[(j+1)%len(shape2)]))])   
    return set(tuple(i) for i in result)

def compute_normals_dict(figure):
    normals_dic = {}
    for i in range(len(figure)):
        for j in range(len(figure)):
            if(i != j):
                normals_dic[i,j]= compute_normals_2_shapes(figure[i], figure[j])
    return normals_dic

def computeDirections(piece, randomness = False): 
    normalsDictionnary = compute_normals_dict(piece)
    directions = []
    r,s=1,1
    if(randomness):
        r,s = np.random.uniform(0.7, 1.3), np.random.uniform(0.7,1.3)
    for i in range(len(piece)):
        directionsi = []

        for key in normalsDictionnary:
            if(key[0]==i and len(normalsDictionnary[key]) > 0) :
                directionsi.append(np.array((list(normalsDictionnary[key])[0])))
        directions.append(directionsi)

    for i in range(len(directions)):
        directions[i] = (r *directions[i][0]+ s * directions[i][1])/ np.linalg.norm(r *directions[i][0]+ s * directions[i][1])
    return directions


baseGridArr =700* np.array([([(i,j),(i+1, j),(i+1,j+1),(i,j+1)]) for i in range(4) for j in range(4)])

def isNeighboorsOf(shape1, shape2):
    #simpleDisplay(np.array([shape1,shape2]))
    for i in range(len(shape1)):
        for j in range(len(shape2)):
            if(isTouching(shape1[i],shape1[(i+1)%len(shape1)],shape2[j],shape2[(j+1)%len(shape2)])):
                return True
    return False
def isNeighboorsVertical(shape1, shape2):
    #simpleDisplay(np.array([shape1,shape2]))
    for i in range(len(shape1)):
        for j in range(len(shape2)):
            if(isTouching(shape1[i],shape1[(i+1)%len(shape1)],shape2[j],shape2[(j+1)%len(shape2)])):
                if(shape1[i][0]==shape1[(i+1)%len(shape1)][0]):
                    return True
    return False
def generateGridGraph(grid):
    G = nx.Graph()
    for i in range(len(grid)):
        G.add_node(i)
    for i in range(len(grid)):
        for j in range(len(grid)):
            if(i!=j and isNeighboorsOf(grid[i],grid[j])):
                G.add_edge(i,j)
                G.add_edge(j,i)
    return G


graph = generateGridGraph(baseGridArr)


def getCycles(graph, grid):
    cycles = nx.minimum_cycle_basis(graph)
    finalCycles = [cycles[0]]
    append = True
    for i in range(len(cycles)): 
        append = True
        for j in range(len(finalCycles)):
            if (intersection(cycles[i], finalCycles[j])!= []):
                append=False        
        if(append and cycles[i] not in finalCycles ):
            finalCycles.append(cycles[i])
    finalCycles= [ rearangeCycle(finalCycles[i], grid)  for i in range(len(finalCycles)) ]
    return finalCycles  

def rearangeCycle(cycle, grid):
    temp = [cycle[0]]
    for i in range(len(cycle)):
        for j in range(len(cycle)):
            if(isNeighboorsOf(grid[temp[i]], grid[cycle[j]])):
                if(cycle[j] not in temp):
                    temp.append(cycle[j])
                    break
    return temp
    
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 
def intersection2DArray(a1, a2):
    temp = []
    for i in range(len(a1)):
        for j in range(len(a2)):
            
            if (np.allclose(a1[i],a2[j])):
                temp.append(a1[i])
    return np.array(temp)
def correctPoints(v1):
    v1 = np.round(np.flip(v1, 0),decimals=3)
    _, idx = np.unique(v1,axis = 0, return_index=True)
    return v1[np.sort(idx)]


def designJoint2parts(p1,p2,v, width, height, distance_to_center=0.5 ):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    v = v/ np.linalg.norm(v)
    nrows, ncols = p1.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [p1.dtype]}

    commonPoint = intersection2DArray(p1,p2)

    JointAxis = np.asarray((commonPoint[1]-commonPoint[0])/np.linalg.norm(commonPoint[1]-commonPoint[0]))

    

    midPoint = np.asarray((commonPoint[0]+commonPoint[1])*distance_to_center)
    alpha = math.acos(np.dot(v,JointAxis)/(np.linalg.norm(v)*np.linalg.norm(JointAxis)))
    polyJoint = shy.Polygon([(midPoint+ JointAxis*width), (midPoint + JointAxis*width + height/math.sin(alpha)*v),(midPoint - JointAxis*width + height/math.sin(alpha)*v),(midPoint - JointAxis*width ),(midPoint - JointAxis*width - height/math.sin(alpha)*v),(midPoint + JointAxis*width - height/math.sin(alpha)*v)])
    #simpleDisplay(np.array([p1]))
    #print(alpha)
    #simpleDisplay(np.array(list([polyJoint.exterior.coords])))

    p1 = shy.Polygon(p1).difference(polyJoint)
    p2 = shy.Polygon(p2).union(polyJoint)
    
    return correctPoints(list(p1.exterior.coords)), correctPoints(list(p2.exterior.coords))


def onePieceJD(piece,directionIndex, randomness=False):
    width = 100
    height = 100
    piece = list(piece)
    

    directions  = computeDirections(piece, randomness)
    for i in range(len(piece)):
        #print( designJoint2parts(piece[i],piece[(i+1)%len(piece)],directions[i], width, height, distance_to_center=0.5 ))
        piece[i], piece[(i+1)%len(piece)] = designJoint2parts(piece[i],piece[(i+1)%len(piece)],directions[i], width, height, distance_to_center=0.5 )
    
    return piece    
def gridDesign(grid, randomness = False):
    width = 100
    height = 100
    print(randomness) 
    graph = generateGridGraph(grid)
    cycles = getCycles(graph, grid)
    groupsOfPieces = []
    pieceLeastNeighborsIndex = []
    for i in range(len(cycles)):
        temp = []
        minNei = 0
        dij = 100
        for j in range(len(cycles[i])):
            temp.append(grid[cycles[i][j]])
            if(len([n for n in graph.neighbors(cycles[i][j])]) < dij):
                minNei = j
                dij = len([n for n in graph.neighbors(cycles[i][j])])
        groupsOfPieces.append(temp)
        pieceLeastNeighborsIndex.append(minNei)
    groupsOfPieces = np.array(groupsOfPieces)
    cutPieces = []
    for i in range(len(groupsOfPieces)):
        cutPieces.append(onePieceJD(groupsOfPieces[i],pieceLeastNeighborsIndex[i], randomness))
        
    asscociatesDirections = np.array([[-1. , 0.],[-1.,  1.], [0., -1.],[1.,  1.] ])
       
    for i in range(len(cutPieces)):
        for j in range(len(cutPieces)):
            
            
            for k in range(len(cutPieces[i])):
                for l in range(k,len(cutPieces[j])):
                    if(i!= j and k!=l):
                        if(isNeighboorsOf(cutPieces[i][k], cutPieces[j][l])):
                            cutPieces[i][k], cutPieces[j][l] = designJoint2parts(cutPieces[i][k], cutPieces[j][l], (asscociatesDirections[i] -asscociatesDirections[j]) / np.linalg.norm(asscociatesDirections[i]+asscociatesDirections[j]), width, height )


    finalPiece = []                        
    for i in range(4):
        for j in range(4):
            finalPiece.append(cutPieces[i][j])
    

    fourth = []
    for i in range(4):
        tempi = shy.Polygon(finalPiece[i*4])
        for j in range(3):
            tempi = tempi.union(shy.Polygon(finalPiece[i*4 + j +1]))
        fourth.append(np.array(list(tempi.exterior.coords)))
        
    simpleDisplay(np.array(finalPiece))
    return finalPiece

piece = gridDesign(baseGridArr, randomness = True)

