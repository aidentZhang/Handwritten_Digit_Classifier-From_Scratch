import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import keras




# neuron numbering starts at 0, layer numbering starts at 0, id starts at 0
class node:
        def __init__(self, layer, neuron, value, id, bias, preSigValue):
                self.layer = layer
                self.neuron = neuron
                self.value = value
                self.preSigValue = preSigValue
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

def sigmoid_Derivative(x):
        return sigmoid(x)*(1-sigmoid(x))

def updateGraph():
        plt.clf()
        nx.draw(G, pos = pos, font_weight='ultralight')
        nx.draw_networkx_labels(G, pos, labels, font_color = "yellow", font_size = 10)
        plt.pause(0.2)

#LAYERS START FROM ZERO, pass in a list of the first layer of neurons in form [[neuron, [connection, connection]], [neuron...]]
def forwardPropagate(LayerValues, layer, network, labels):
        if layer == 0:
                i=0
                while i < len(LayerValues):
                        if i < 5 or i > 779: 
                                labels[i] = LayerValues[i]
                        i+=1
                updateGraph()


        if layer == num_layers-1:
                return LayerValues, network
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
        


        #update graph
        i = 0
        neuronsSoFar = 0
        while i <= layer:
                neuronsSoFar+=nodesPerLayer[i]
                i+=1
        i=0
        while i < len(temp):
                #CHECK THIS LINE
                network[layer+1][i][0].value = sigmoid(temp[i])
                network[layer+1][i][0].preSigValue = temp[i]

                labels[neuronsSoFar+i] = round(sigmoid(temp[i]), 2)
                i+=1
        updateGraph()

        


        return forwardPropagate(temp, layer+1, network, labels)


#INITIALIZE NETWORK STRUCT
num_layers = 4
nodesPerLayer = [784,16,16,10]








network = []

i = 0
#print("hello")
ids = 0
while i < num_layers:
        temp = []
        j = 0
        while j < nodesPerLayer[i]:
                temp.append([node(i, j, 1, ids, 0, 0)])
                j+=1
                ids+=1
        network.append(temp)
        i+=1
i = 0




#SPACING INFO
xSpace = float(2)/(ids+2)
ySpace = float(2)/(16 if max(nodesPerLayer) > 16 else max(nodesPerLayer))
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




G = nx.complete_multipartite_graph()

connections = []
i = 0
while i < num_layers-1:
        for curr in network[i]:
                temp = []         

                for next in network[i+1]:
                        temp.append(connection(random.random()*2-1, i, curr[0].neuron, next[0].neuron, curr[0].id, next[0].id))
                        if curr[0].id >779 or curr[0].id < 5:
                                G.add_edge(curr[0].id, next[0].id)
                network[i][curr[0].neuron].append(temp)
        i+=1



plt.ion()
labels = {}

i = 0
while i < ids:
        labels[i] = 0
        i+=1
i = 0

#layers start from 0, calculate 
def find_Gradient(layer, updateList, expected_outcomes):
        if layer == 0:
                i = 0
                while i < len(updateList[0]):
                        updateList[0][i][0] = 0
                        i+=1
                return updateList
        elif layer == num_layers-1:
                #calculate gradient for weights first
                for neuron_Group in network[layer-1]:
                        #print("sdad")
                        #print(neuron_Group)
                        for link in neuron_Group[1]:

                                current_PreSigVal = network[layer][link.toNeuron][0].preSigValue
                                #print(sigmoid_Derivative(current_PreSigVal))
                                #print("messy")
                                #print(neuron_Group[0])
                                updateList[layer-1][link.fromNeuron][1][link.toNeuron] = (sigmoid(current_PreSigVal)-expected_outcomes[link.toNeuron])*2*sigmoid_Derivative(current_PreSigVal)*neuron_Group[0].value
                #calculate gradient for bias
                for neuron_Group in network[layer]:
                        current_PreSigVal = network[layer][link.toNeuron][0].preSigValue
                        updateList[layer][neuron_Group[0].neuron][0] = 2*(sigmoid(current_PreSigVal)-expected_outcomes[neuron_Group[0].neuron])
        #Processing for cases in between first and last layer REMEBER THAT EACH NEURON INFLUENCES COST THROUGH MULTIPLE PATHS
        
        
        else:
                for neuron_Group in network[layer-1]:
                        for link in neuron_Group[1]:
                                delC_delA = 0
                                i = 0
                                while i < len(updateList[layer][link.toNeuron][1]):
                                        delC_delA+=updateList[layer][link.toNeuron][1][i]*(1/(network[layer][link.toNeuron][0].value))*network[layer][link.toNeuron][1][i].weight
                                        i+=1
                                delC_delA*=(sigmoid_Derivative(network[layer][link.toNeuron][0].preSigValue)*network[layer-1][link.fromNeuron][0].value)
                                updateList[layer-1][link.fromNeuron][1][link.toNeuron] = delC_delA
                
                #calculate gradient for bias


                for neuron_Group in network[layer]:
                        i = 0
                        delC_delA = 0
                        while i < len(network[layer+1]):
                                delC_delA+=(updateList[layer+1][i][0] * neuron_Group[1][i].weight)
                                i+=1

                        updateList[layer][neuron_Group[0].neuron][0] = delC_delA
        

                
        return find_Gradient(layer-1, updateList, expected_outcomes)
        

def updateNetwork(network, updateList):
        i = 0
        while i < len(network):
                j = 0
                while j < len(network[i]):
                        network[i][j][0].bias -= updateList[i][j][0]*growth_Factor
                        if i != len(network)-1:
                                k = 0
                                while k < len(network[i][j][1]):
                                        network[i][j][1][k].weight -= updateList[i][j][1][k]*growth_Factor
                                        k+=1
                        j+=1
                i+=1
        return network

growth_Factor = 0.3
import copy
#print(network)
updateList = copy.deepcopy(network)






#print(answer)
import pdb
(x_preprocessed, y_preprocessed), (x_test, y_preprocessed) = keras.datasets.mnist.load_data()
#print(x_train[0])
x_preprocessed = x_preprocessed[:500]
y_preprocessed = y_preprocessed[:500]
x_train = []
x = 0
while x < len(x_preprocessed):
        j = 0
        temp = []
        while j < 28:
                k = 0
                while k < 28:
                        temp.append(float(int(x_preprocessed[x][j][k]))/255)
                        k+=1
                j+=1
        x+=1
        x_train.append(temp)
y_train = []
x = 0
while x < len(y_preprocessed):
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[y_preprocessed[x]] = 1
        y_train.append(temp)
        x+=1

print(len(x_train[0])) 
repeats = 0
while repeats < 500:
        answer, network = forwardPropagate(x_train[repeats], 0, network, labels)
        gradient= find_Gradient(num_layers-1, updateList, y_train[repeats])

        network = updateNetwork(network, gradient)
        repeats+=1

plt.ioff()
plt.show()


