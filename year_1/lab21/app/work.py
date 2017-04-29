#coding: utf-8
import itertools
import copy
from flask import render_template
from flask import request
from app import app
import json
import sys

#Method Magu

def magu(Number_of_vertices, List_of_undirected_edges):
    print(Number_of_vertices, file=sys.stderr)
    print(List_of_undirected_edges, file=sys.stderr)
    Set_of_all_subsets = list(itertools.product(*List_of_undirected_edges))
    Selection_of_independent_subsets = list(set( [tuple(set(p)) for p in Set_of_all_subsets] ))
    Selection_of_independent_subsets = list(map(set, Selection_of_independent_subsets))
    Selection_of_independent_subsets.sort(key = len)
        
    deleted_of_subsets = [False]*len(Selection_of_independent_subsets)

    for i in range(len(Selection_of_independent_subsets)):
        for j in range(i + 1, len(Selection_of_independent_subsets)):
            if not deleted_of_subsets[j] and Selection_of_independent_subsets[i] <= Selection_of_independent_subsets[j]:
                deleted_of_subsets[j] = True

    fullSet = set([i for i in range(Number_of_vertices)]) 
    max_subsets = [Selection_of_independent_subsets[i] for i in range(len(Selection_of_independent_subsets)) if not deleted_of_subsets[i]]
    max_independent_subsets = [fullSet - s for s in max_subsets]
        
    #The method Magoo reduces the coloring task to transformations of Boolean functions
        
    Psi = []
    for v in range(Number_of_vertices):
        dis_y = []
        for i in range(len(max_subsets)):
            if v not in max_subsets[i]:
                dis_y.append(i)    
        Psi.append(dis_y)
        
    #Similar implementation as in the method of Magu    
        
    Psi2 = list(itertools.product(*Psi))
    Psi3 = list(set( [tuple(set(p)) for p in Psi2] ))
    Psi3 = list(map(set, Psi3))
    Psi3.sort(key = len)
        
    deleted = [False]*len(Psi3)

    for i in range(len(Psi3)):
        for j in range(i+1, len(Psi3)):
            if not deleted[j] and Psi3[i] <= Psi3[j]:
                deleted[j] = True

    #Coverings of a graph by maximal independent sets
    Psi4 = [Psi3[i] for i in range(len(Psi3)) if not deleted[i]]   
    
    #selection min сoatings    
    gamma = min( [ len(elem) for elem in Psi4] )
    Psi5 = [list(elem) for elem in Psi4 if len(elem) == gamma]
        
        
    # Methods of coloring - there are as many as elements in Psi5
    all_colors = []
    for psi in Psi5:
        colors = [None] * Number_of_vertices
        for i in range(len(psi)):
            startSet = max_independent_subsets[psi[i]]
            for j in range(i):
                startSet = startSet - max_independent_subsets[psi[j]]
            for v in startSet:
                colors[v] = i
        all_colors.append(colors)
        
    return all_colors
    

@app.route('/')
@app.route('/index',methods=['GET', 'POST'])
def index():
    return render_template("beta.html")

@app.route('/cytoscape-edge.js',methods=['GET', 'POST'])
def cytoscape():
    return render_template("cytoscape-edge.js")

@app.route('/code.js', methods=['GET', 'POST'])
def code():
    return render_template("code.js")
        
@app.route('/_find', methods=['GET', 'POST'])
def find():
    data = request.json
    vertex_ids = []
    list_of_undirected_edges = []
    try:
        for i in data['elements']['nodes']:
            vertex_ids.append(i['data']['id'])
    except:
        pass
    try:
        for i in data['elements']['edges']:
            source = vertex_ids.index(i['data']['source'])       # Translate text names of vertices into sequence numbers
            target = vertex_ids.index(i['data']['target'])
            list_of_undirected_edges.append([source, target])
            list_of_undirected_edges.append([target, source])
    except:
        print(sys.exc_info(), file=sys.stderr)

    if request.method == "POST":
        try:
            return json.dumps({'vertex_ids': vertex_ids, 'allColors': magu(len(vertex_ids), list_of_undirected_edges)})
        except:
            print(sys.exc_info(), file=sys.stderr)
    else:
        return -1
