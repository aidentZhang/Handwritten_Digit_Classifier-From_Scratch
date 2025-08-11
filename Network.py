import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import numpy as np
import keras
import copy
import math

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


def addTwo(l1, l2):
        first = 0
        while first<len(l1):
                second = 0
                while second<len(l1[first]):
                        l1[first][second][0] += l2[first][second][0]
                        if len(l1[first][second])>1:
                                third = 0
                                while third<len(l1[first][second][1]):
                                        l1[first][second][1][third] += l2[first][second][1][third]
                                        third+=1
                        second+=1
                first+=1
        return l1

def networkListDiv(l1, x):
        first = 0
        while first<len(l1):
                second = 0
                while second<len(l1[first]):
                        l1[first][second][0] = l1[first][second][0]/x
                        if len(l1[first][second])>1:
                                third = 0
                                while third<len(l1[first][second][1]):
                                        l1[first][second][1][third] = l1[first][second][1][third]/x
                                        third+=1
                        second+=1
                first+=1
        return l1

def tanh(x):
        return 2*sigmoid(2*x)-1

def tanh_derivative(x):
        return 1-tanh(x)**2
def sigmoid(x):
        try:
                return 1/(1+ math.exp(-x))
        except:
                print("poo")
                z = input()

def sigmoid_Derivative(x):
        return sigmoid(x)*(1-sigmoid(x))

def updateGraph():
        if isUpdate:
                plt.clf()
                nx.draw(G, pos = pos, font_weight='ultralight')
                
                nx.draw_networkx_labels(G, pos, labels, font_color = "yellow", font_size = 10)
                
                plt.pause(0.05)

#LAYERS START FROM ZERO, pass in a list of the first layer of neurons in form [[neuron, [connection, connection]], [neuron...]]
def forwardPropagate(LayerValues, layer, network, labels):
        

        if layer == 0:
                i=0
                while i < len(LayerValues):
                        labels[i] = LayerValues[i]
                        network[0][i][0].value = LayerValues[i]
                        i+=1
                        
                n = lowerBound
                while n <= upperBound:
                        del labels[n]
                        n+=1
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
                network[layer+1][i][0].value = tanh(temp[i]) if layer != num_layers - 2 else sigmoid(temp[i])
                network[layer+1][i][0].preSigValue = temp[i]
                temp[i] = tanh(temp[i]) if layer != num_layers - 2 else sigmoid(temp[i])
                labels[neuronsSoFar+i] = round(temp[i], 2)
                i+=1
        updateGraph()
        return forwardPropagate(temp, layer+1, network, labels)



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
                        for link in neuron_Group[1]:
                                current_PreSigVal = network[layer][link.toNeuron][0].preSigValue
        #CHANING THIS TO CROSS ENTROPY DERIVATIVE
                                #updateList[layer-1][link.fromNeuron][1][link.toNeuron] = ((sigmoid(current_PreSigVal)-expected_outcomes[link.toNeuron]))*2*sigmoid_Derivative(current_PreSigVal)*neuron_Group[0].value
#NEW LINE
                                updateList[layer-1][link.fromNeuron][1][link.toNeuron] = ((sigmoid(current_PreSigVal)-expected_outcomes[link.toNeuron]))*neuron_Group[0].value
                #calculate gradient for bias
                for neuron_Group in network[layer]:
#MADE CHANGE HERE
                        current_PreSigVal = neuron_Group[0].preSigValue
                        updateList[layer][neuron_Group[0].neuron][0] = (sigmoid(current_PreSigVal)-expected_outcomes[neuron_Group[0].neuron])

                        #bottom line is the MSE loss derivative thing
                        #updateList[layer][neuron_Group[0].neuron][0] = 2*(sigmoid(current_PreSigVal)-expected_outcomes[neuron_Group[0].neuron])*sigmoid_Derivative(current_PreSigVal)
        #Processing for cases in between first and last layer REMEBER THAT EACH NEURON INFLUENCES COST THROUGH MULTIPLE PATHS
        else:
                for neuron_Group in network[layer-1]:
                        for link in neuron_Group[1]:
                                delC_delA = 0
                                i = 0
                                while i < len(updateList[layer][link.toNeuron][1]):
                                        #delC_delA+=updateList[layer][link.toNeuron][1][i]*network[layer][link.toNeuron][1][i].weight
#rolled back
                                        delC_delA+=updateList[layer][link.toNeuron][1][i]*(1/(network[layer][link.toNeuron][0].value))*network[layer][link.toNeuron][1][i].weight
                                        i+=1
                                delC_delA*=(tanh_derivative(network[layer][link.toNeuron][0].preSigValue)*network[layer-1][link.fromNeuron][0].value)
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




#INITIALIZE NETWORK STRUCT
num_layers = 4
nodesPerLayer = [784, 24, 24, 10]
network = []

i = 0
ids = 0
#NODE CREATION
while i < num_layers:
        temp = []
        j = 0
        while j < nodesPerLayer[i]:
                temp.append([node(i, j, 0, ids, 0, 0)])
                j+=1
                ids+=1
        network.append(temp)
        i+=1


#Info for cropping out nodes in first layer due to spacing constraints
lowerBound = 5
upperBound = 779
#SPACING INFO
xSpace = float(2)/(ids+2)
ySpace = float(2)/(16 if max(nodesPerLayer) > 16 else max(nodesPerLayer))
pos = {}
currX = -1+xSpace
i = 0
#POSITIONING
for layer in nodesPerLayer:
        j = 0
        currY = -1+ySpace

        while j < layer:
                if i < lowerBound or i > upperBound:
                        pos[i] = (currX, currY)
                        currY+=ySpace

                i+=1
                j+=1
        currX+=xSpace



#ADDING CONNECTIONS
G = nx.complete_multipartite_graph()

connections = []
i = 0
while i < num_layers-1:
        for curr in network[i]:
                temp = []         

                for next in network[i+1]:
                        temp.append(connection(random.random()*2 - 1, i, curr[0].neuron, next[0].neuron, curr[0].id, next[0].id))
                        G.add_edge(curr[0].id, next[0].id)
                network[i][curr[0].neuron].append(temp)
        i+=1


#labeling and a bit of housekeeping
labels = {}
i = 0
while i < ids:
        labels[i] = 0
        i+=1
i = 0
growth_Factor = 2           
isUpdate = False

updateList = copy.deepcopy(network)
plt.ion()

# LOAD DATA

# d = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
# a = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

# k = 0
# while k<200:
#         print("Sss")
#         answer, network = forwardPropagate(d[k%4], 0, network, labels)
#         average = find_Gradient(num_layers-1, updateList, a[k%4])
#         network = updateNetwork(network, average)
#         k+=1

#         answer, network = forwardPropagate(d[0], 0, network, labels)


(x_preprocessed, y_preprocessed), (x_test_preprocessed, y_test_preprocessed) = keras.datasets.mnist.load_data()
x_preprocessed = x_preprocessed[:5000]
y_preprocessed = y_preprocessed[:5000]
x_train = []
x = 0
#PROCESS DATA

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
l = 0
while l < len(y_preprocessed):
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[y_preprocessed[l]] = 1
        y_train.append(temp)
        l+=1

x=0
#Test data processing
x_test = []
while x < len(x_test_preprocessed):
        j = 0
        temp = []
        while j < 28:
                k = 0
                while k < 28:
                        temp.append(float(int(x_test_preprocessed[x][j][k]))/255)
                        k+=1
                j+=1
        x+=1
        x_test.append(temp)
y_test = []
l = 0
while l < len(y_test_preprocessed):
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[y_test_preprocessed[l]] = 1
        y_test.append(temp)
        l+=1




i = lowerBound
while i <= upperBound:
        G.remove_node(i)
        i+=1

from tqdm import tqdm

#TRAIN
totalEpoches = 5



o = 0

# while True:
#         answer, network = forwardPropagate(x_train[0], 0, network, labels)
#         average = find_Gradient(num_layers-1, updateList, y_train[0])
#         network = updateNetwork(network, average)


#Testing
# # for i in network:
# #         for j in i:
# #                 print(j[0].bias)
# #                 try:
# #                         for z in j[1]:
# #                                 print(z.weight, end="")
# #                 except:
# #                         continue

# # while o < totalEpoches:
# #         repeats = 0
# #         correct = 0
# #         with tqdm(total=len(x_train)) as pbar:
# #                 while repeats < len(x_train):
# #                         answer, network = forwardPropagate(x_train[repeats], 0, network, labels)
# #                         if o == 2:
# #                                 print("Epoch: "+ str(o)+ " Pass number " + str(repeats) + " Expected: " + str(y_train[repeats].index(max(y_train[repeats])))+", got " + str(answer.index(max(answer))))
# #                         if y_train[repeats].index(max(y_train[repeats])) == (answer.index(max(answer))):
# #                                 correct+=1
# #                         gradient = find_Gradient(num_layers-1, updateList, y_train[repeats])
# #                         pbar.update(1)
# #                         network = updateNetwork(network, gradient)
# #                         repeats+=1
# #         print("Epoch: "+ str(o) + " done now accuracy = " + str(correct/len(x_train)))
# #         o+=1
# # x_train = [[0,0,1,1],[0,1,0,1],[1,1,0,0],[1,0,1,0]]
# # y_train = [[1,0], [0,1], [1,0], [0,1]]

def find_Loss(answer, target):
        x = 0.0
        z = 0
        while z < len(answer):
                x+= ((answer[z]-target[z])**(2))
                z+=1
        return x



batch_Size = 50
                
testingtemp = 0

while o < totalEpoches:
        curr = 0
        with tqdm(total=len(x_train)) as pbar:
                while curr< len(x_train):
                        left = batch_Size if (len(x_train)-curr)/batch_Size >= 1.0 else len(x_train)%batch_Size
                        y = 0
                        loss = 0
                        answer, network = forwardPropagate(x_train[curr], 0, network, labels)
                        average = find_Gradient(num_layers-1, updateList, y_train[curr])
                        loss += find_Loss(answer, y_train[curr])
                        curr+=1
                        y+=1
                        while y < left:    #Batch size of 64
                                try:
                                        answer, network = forwardPropagate(x_train[curr], 0, network, labels)
                                except:
                                        print(f"{curr}.  {left}.   {y}")
                                        input()
                                temp = find_Gradient(num_layers-1, updateList, y_train[curr])
                                average = addTwo(average, temp)
                                loss += find_Loss(answer, y_train[curr])

                                y+=1
                                curr+=1
                                pbar.update(1)
                                
                        loss/=batch_Size
                        print(f"Loss = {loss}")
                        average = networkListDiv(average, batch_Size)
                        network = updateNetwork(network, average)        
        i = 0
        correct = 0
        while i < 100:
                answer, network = forwardPropagate(x_train[i], 0, network, labels)
                print(f"expected {y_train[i]} got {answer}")
                if y_train[i].index(1) == answer.index(max(answer)):
                        print("correct!")
                        correct+=1
                i+=1
                #Growth factor shrinking
        print(f"Correct proportion is {correct/100}")
        #growth_Factor * (o/(o+1))
                
        print("Epoch done")
        o+=1


i = 0
correct = 0
while i < len(x_test):
        answer, network = forwardPropagate(x_test[i], 0, network, labels)
        print(f"expected {y_test[i]} got {answer}")
        if y_test[i].index(1) == answer.index(max(answer)):
                correct+=1
        i+=1
print(correct/len(x_test))


plt.ioff()
plt.show()


