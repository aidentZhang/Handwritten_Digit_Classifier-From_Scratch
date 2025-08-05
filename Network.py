import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import numpy as np
class node:
        def __init__(self, layer, neuron, value, id, bias):
                self.layer = layer
                self.neuron = neuron
                self.value = value
                self.id = id
                #bias is the bias added onto the connection going to it. First layer has no bias

                self.bias = bias

class connection:
        def __init__(self, weight, fromLayer, fromNeuron, toNeuron, fromId, toId):
                self.weight = weight
                self.fromLayer = fromLayer
                self.fromNeuron = fromNeuron
                self.toNeuron = toNeuron
                self.fromId = fromId
                self.toId = toId

def sigmoid(x):
       return 1/(1+ 2.71828**(-x))
print(sigmoid(2.498671211709183))
#LAYERS START FROM ZERO, pass in a list of the first layer of neurons in form [[neuron, [connection, connection]], [neuron...]]
def forwardPropagate(LayerValues, layer, network, labels):
        if layer == 0:
            i=0
            while i < len(LayerValues):
                    labels[i] = LayerValues[i]
                    i+=1
            plt.clf()
            nx.draw(G, pos = pos, font_weight='ultralight')

            nx.draw_networkx_labels(G, pos, labels, font_color = "orange", font_size = 20)
            plt.pause(5)

    
        if layer == num_layers-1:
            return LayerValues
        temp = []
        i = 0
        for neuron in network[layer+1]:
               temp.append(neuron[0].bias)
        curr = 0
        for neuron in network[layer]:
            j = 0
            for connection in neuron[1]: 
                temp[j]+=connection.weight*LayerValues[curr]
                j+=1
            curr+=1
        i = 0
        while i < len(temp):
               temp[i] = sigmoid(temp[i])
               i+=1


        #update graph
        i = 0
        neuronsSoFar = 0
        while i <= layer:
               neuronsSoFar+=nodesPerLayer[i]
               i+=1
        print(neuronsSoFar)
        i=0
        while i < len(temp):
                labels[neuronsSoFar+i] = round(temp[i], 2)
                i+=1
        print(labels)
        plt.clf()
        nx.draw(G, pos = pos, font_weight='ultralight')

        nx.draw_networkx_labels(G, pos, labels, font_color = "orange", font_size = 20)
        plt.pause(2)

        


        return forwardPropagate(temp, layer+1, network, labels)


#INITIALIZE NETWORK STRUCT
num_layers = 5
nodesPerLayer = [7, 5, 5, 5, 5]








network = []

i = 0
#print("hello")
ids = 0
while i < num_layers:
        temp = []
        j = 0
        while j < nodesPerLayer[i]:
                temp.append([node(i, j, 1, ids, 1)])
                j+=1
                ids+=1
        network.append(temp)
        i+=1
i = 0




#SPACING INFO
xSpace = float(2)/(ids+2)
ySpace = float(2)/(max(nodesPerLayer))
pos = {}
currX = -1+xSpace
i = 0

for layer in nodesPerLayer:
        j = 0
        currY = -1+ySpace

        while j < layer:
                pos[i] = (currX, currY)
                i+=1
                currY+=ySpace
                j+=1
        currX+=xSpace




#print(network)
G = nx.complete_multipartite_graph()

connections = []
i = 0
while i < num_layers-1:
        for curr in network[i]:
                temp = []         

                for next in network[i+1]:
                        #print("here")
                        temp.append(connection(random.random()*2, i, curr[0].neuron, next[0].neuron, curr[0].id, next[0].id))
                        G.add_edge(curr[0].id, next[0].id)
                        #print("edge added")
                #print(temp)
                network[i][curr[0].neuron].append(temp)
        i+=1


#print(network)

plt.ion()
labels = {}

i = 0
while i < ids:
       labels[i] = 0
       i+=1
i = 0


print(forwardPropagate([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0, network, labels))







plt.ioff()
plt.show()


