import math as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import random as rd

def distanceEu(a, b):
    d = 0.0
    for i in range(len(a)):
        d += (b[i]-a[i])*(b[i]-a[i])
            
    return np.sqrt(d)

def neighborhoodFunction(dist, sigma):
    return m.exp((-dist**2)/(2*sigma**2))

def distanceManh(bmu, vector):
    """ Manhattan distance calculation of coordinates

    :param bmu: {numpy.ndarray} array / vector
    :param vector: {numpy.ndarray} array / vector
    :param shape: {tuple} shape of the SOM
    :return: {numpy.ndarray} Manhattan distance for v to m
    """

    delta = np.abs(bmu - vector)

    return np.sum(delta, axis=len(bmu.shape) - 1)


class SCCSOM_EAC_SL(object):
    def __init__(self, shape, alpha_start=0.6, seed=None):
        """ Initialize the SOM object with a given map size
        
        :param +
        : {int} width of the map
        :param y: {int} height of the map
        :param alpha_start: {float} initial alpha at training start
        :param seed: {int} random seed to use
        """
        np.random.seed(seed)
        self.shape = shape
        self.dim = len(shape)
        self.sigma = shape[0] / 2.
        self.alpha_start = alpha_start
        self.alphas = None
        self.sigmas = None
        self.epoch = 0
        self.interval = 0
        self.map = np.array([])
        self.distmap = np.zeros(shape)
        self.winner_indices = np.array([])
        self.pca = None  # attribute to save potential PCA to for saving and later reloading
        self.inizialized = False
        self.error = 0.  # reconstruction error
        self.history = []  # reconstruction error training history
        self.historyInf = []  # reconstruction error training history
        self.clusters = []
        self.numML = []
        self.dataInSOM = []
        self.dataInCluster = []
        self.infeasibilityEpoch = []
        self.distICEpoch = []
        self.mapWinner = np.full(shape,  list)
        self.mapCentroides = np.full(shape,  list)
        self.mapVariance = np.zeros(shape)
        
        n_neuronas = 1
        for i in range(len(shape)):
            n_neuronas *= shape[i]
            
        aux = n_neuronas
        self.pos = np.zeros(self.dim)
        for i in range(len(self.pos)):
            aux /= shape[i]
            self.pos[i] = aux

        self.winners = np.full(n_neuronas,  list)
        self.numNeuronas = n_neuronas
        self.indxmap = np.stack(np.unravel_index(np.arange(n_neuronas, dtype=int).reshape(shape), shape), self.dim)
        self.centroides = np.full(n_neuronas,  list)
        self.variance = np.zeros(n_neuronas)
        self.varianceTotal = 0

    def initialize(self, pobl, restr):
        """ Initialize the SOM neurons

        :param data: {numpy.ndarray} data to use for initialization
        :param how: {str} how to initialize the map, available: 'pca' (via 4 first eigenvalues) or 'random' (via random
            values normally distributed like data)
        :return: initialized map in self.map
        """
        self.pobl = pobl
        self.buildCoasoaciationMatrix()
        self.sortPobl()

        self.map = np.random.normal(np.mean(pobl), np.std(pobl), size=tuple(np.append(self.numNeuronas, len(pobl[0]))))
                
        # Calcular el n??mero de restricciones ML que tiene cada instancia
        self.numML = np.zeros(len(pobl))
        for i in range(len(pobl)):
            for j in range(i+1, len(pobl)):
                if(restr[i][j]==1):
                    self.numML[i]+=1
                    self.numML[j]+=1
                    
        # Inicializar a -1 el cluster al que pertenecen las instancias.
        self.dataInSOM = np.full(len(pobl), -1)            
        self.inizialized = True
        
        # Mapa con las entradas clasificadas
        for i in range(self.numNeuronas):
            self.winners[i]=[]
        
    def neuronaEnLista(neurona):
        posicion = 0
        for i in range(len(neurona)):
            posicion += self.pos[i] * neurona[i]
        return posicion
    
    def incrementInfeasibility(self, data_neurona, ind, restr):
        infeasibility = self.numML[ind]
        
        for i in data_neurona:
            if(i!=ind):
                if(restr[i]==-1):
                    infeasibility+=1
                if(restr[i]==1):
                    infeasibility-=1
            
        return infeasibility;

    def winner(self, vector, indice, restr):
        """ Compute the winner neuron closest to the vector (Euclidean distance)

        :param vector: {numpy.ndarray} vector of current data point(s)
        :return: indices of winning neuron
        """
        mejor_inf = len(restr)
        mejor_d = 1000
        
        for i in range(len(self.winners)):
            inf = self.incrementInfeasibility(self.winners[i], indice, restr)
            if(inf < mejor_inf):
                mejor_inf = inf
                mejor_neurona = i
                mejor_d = distanceEu(vector, self.map[i])
            elif(inf == mejor_inf):
                d = distanceEu(vector, self.map[i])
                if(d < mejor_d):
                    mejor_inf = inf 
                    mejor_neurona = i
                    mejor_d = d
        return mejor_neurona

    def cycle(self, vector, indice, epoch, restr):
        """ Perform one iteration in adapting the SOM towards a chosen data point

        :param vector: {numpy.ndarray} current data point
        """
        neu = self.dataInSOM[indice]
        if(neu!=-1):
            self.winners[neu].remove(indice)
                
        w = self.winner(vector, indice, restr)
        
        self.winners[w].append(indice)
        self.dataInSOM[indice] = w
        
        v = np.unravel_index(w, self.shape)
        
        for i in range(len(self.map)):
            # get Manhattan distance of every neuron in the map to the winner
            j = np.unravel_index(i, self.shape)
            dist = distanceManh(self.indxmap[v], self.indxmap[j])
        
            # smooth the distances with the current sigma
            h = neighborhoodFunction(dist, self.sigmas[self.epoch])
            
            # update neuron weights
            self.map[i] += h * self.alphas[self.epoch] * (vector - self.map[i])

    def fit(self, restr, epochs=0, save_e=False, decay='hill'):
        """ Train the SOM on the given data for several iterations

        :param data: {numpy.ndarray} data to train on
        :param epochs: {int} number of iterations to train; if 0, epochs=len(data) and every data point is used once
        :param save_e: {bool} whether to save the error history
        :param decay: {str} type of decay for alpha and sigma. Choose from 'hill' (Hill function) and 'linear', with
            'hill' having the form ``y = 1 / (1 + (x / 0.5) **4)``
        """
        
        indices = np.arange(len(self.pobl))
        self.infeasibilityEpoch = np.zeros(epochs)
        self.distICEpoch = np.zeros(epochs)
        
        # get alpha and sigma decays for given number of epochs or for hill decay
        if decay == 'hill':
            epoch_list = np.linspace(0, 1, epochs)
            self.alphas = self.alpha_start / (1 + (epoch_list / 0.5) ** 4)
            self.sigmas = self.sigma / (1 + (epoch_list / 0.5) ** 4)
        else:
            self.alphas = np.linspace(self.alpha_start, 0.05, epochs)
            self.sigmas = np.linspace(self.sigma, 1, epochs)

        lista = constraintsList(restr)
        for i in range(epochs):
            for j in indices:
                self.cycle(self.pobl[j], j, i, restr[j])
            self.distICEpoch[self.epoch] = self.calcularDistICTotal(mode="som")
            self.infeasibilityEpoch[self.epoch] = self.calcularInfeasibilityTotal(constraintsList(restr), mode="som")
            self.epoch = self.epoch + 1
            self.historyInf.append(self.calcularInfeasibilityTotal(lista, mode="som"))
        
        self.build_map_clusters()

    def som_error(self, data):
        """ Calculates the overall error as the average difference between the winning neurons and the data points

        :param data: {numpy.ndarray} data matrix to calculate SOM error
        :return: normalized error
        """
        error = 0.0
        distIC = np.zeros(len(self.map))
        
        for i in range(len(self.map)):
            for j in self.winners[i]:
                distIC[i] += distanceEu(data[j],self.map[i])
            if(len(self.winners[i]) > 0):
                distIC[i] /= len(self.winners[i])
            error += distIC[i]
        
        return error/len(self.map)

    def generateKClusters(self, restr, n):
        while(len(self.clusters)<n):
            ajuste = False
            for i in range(self.numNeuronas):
            	if(len(self.winners[i])==0):
                    ind = rd.randint(0, len(self.pobl)-1)
                    self.winners[self.dataInSOM[ind]].remove(ind)
                    self.winners[i].append(ind)
                    self.dataInSOM[ind] = i
                    ajuste = True
            if(ajuste):        
                self.build_map_clusters()

        while(len(self.clusters)>n):
            mejorVt = np.Inf
            mejor_inf = np.inf
            existeVecino = False
            
            dist = 1
            cambio = False
            while(not existeVecino):
                for i in range(len(self.clusters)):
                    for j in range(i+1, len(self.clusters)):
                            if(neighbour(self.clusters[i], self.clusters[j], dist)):
                                existeVecino = True
                                inf = self.incrementInfeasibilityIC(restr, self.clusters[i].inputs, self.clusters[j].inputs)
                                if(inf <= mejor_inf):
                                    centroide = (self.clusters[i].centroide * self.clusters[i].numInputs + self.clusters[j].centroide * self.clusters[j].numInputs) / (self.clusters[i].numInputs + self.clusters[j].numInputs)
                                    d = 0
                                    for k in range(self.clusters[i].numInputs):
                                        d += distanceEu(self.pobl[self.clusters[i].inputs[k]], centroide)
                                    for k in range(self.clusters[j].numInputs):
                                        d += distanceEu(self.pobl[self.clusters[j].inputs[k]], centroide)
                                    vt = self.varianceTotal + d - self.clusters[i].variance - self.clusters[j].variance
                                    
                                    if(inf < mejor_inf):
                                        cambio = True
                                        mejor_inf = inf
                                        eli1 = i
                                        eli2 = j
                                        mejorVt = vt
                                        mejorCl = Cluster(self.clusters[i].arrayPos + self.clusters[j].arrayPos, centroide, vt, self.clusters[i].inputs + self.clusters[j].inputs)
                                        
                                    elif(mejor_inf == inf):
                                        if (vt < mejorVt):
                                            cambio = True
                                            eli1 = i
                                            eli2 = j
                                            mejorVt = vt
                                            mejorCl = Cluster(self.clusters[i].arrayPos + self.clusters[j].arrayPos, centroide, vt, self.clusters[i].inputs + self.clusters[j].inputs)
                if(not existeVecino):
                    dist+=1
                else:
                    if(cambio):
                        self.varianceTotal = mejorVt
                        self.clusters.pop(eli2)        
                        self.clusters.pop(eli1)        
                        self.clusters.append(mejorCl)
            
        for i in range(len(self.clusters)):
            for j in self.clusters[i].inputs:
                self.dataInCluster[j] = i
                    
    def calcularInfeasibilityTotal(self, lista, mode="cluster"):
        inf = 0
        if(mode == "som"):
            for i, j, restr in lista:
                if(restr==-1 and self.dataInSOM[i]==self.dataInSOM[j]):
                    inf+=1
                if(restr==1 and self.dataInSOM[i]!=self.dataInSOM[j]):
                    inf+=1
        else:
            for i, j, restr in lista:
                if(restr==-1 and self.dataInCluster[i]==self.dataInCluster[j]):
                    inf+=1
                if(restr==1 and self.dataInCluster[i]!=self.dataInCluster[j]):
                    inf+=1
           
        return inf
    
    def incrementInfeasibilityIC(self, restr, cl1, cl2):
        inf = 0
        for i in cl1:
            for j in cl2:
                if(restr[i,j]==1):
                    inf-=1
                if(restr[i,j]==-1):
                    inf+=1
           
        return inf
    
    def calcularDistICTotal(self, mode="cluster"):
        d = 0.0
        if(mode=="som"):
            for i in range(self.numNeuronas):
                self.centroides[i]=np.zeros(len(self.pobl[0]))
                for j in range(len(self.winners[i])):
                    self.centroides[i]+=self.pobl[self.winners[i][j]]
                if(len(self.winners[i])!=0):
                    self.centroides[i]/=len(self.winners[i])
                    
            for i in range(self.numNeuronas):
                aux = 0
                for j in range(len(self.winners[i])):
                    aux+= distanceEu(self.pobl[self.winners[i][j]], self.centroides[i])
                if(len(self.winners[i])>0):
                    aux /= len(self.winners[i])
                    d += aux
            d /= self.numNeuronas
        else:
            for i in self.clusters:
                d += i.variance/i.numInputs
            
            d /= len(self.clusters)
           
        return d
    
    def buildCoasoaciationMatrix(self):
        coasociacion = np.zeros((len(self.pobl), len(self.pobl)))
        for i in range(len(self.pobl)):
            for j in range(i+1, len(self.pobl)):
                suma = 0
                for k in range(15):
                    if(self.pobl[i][k] == self.pobl[j][k]):
                        suma += 1
                suma /= 15
                coasociacion[i][j] = suma
                coasociacion[j][i] = suma
        self.coasociacion = coasociacion
            
    def sortPobl(self):
        co = self.coasociacion
        for i in range(len(self.pobl)):
            co[i][i] = -1

        poblOrdenada = []
        elegidos = np.zeros(len(self.pobl))
        
        maximo = np.amax(co)
        while(maximo != -1):
            ind = np.unravel_index(np.argmax(co, axis=None), co.shape)
            co[ind[0], :] = -1
            co[ind[1], :] = -1
            
            if(elegidos[ind[0]] == 0):
                poblOrdenada.append(self.pobl[ind[0]])
                elegidos[ind[0]] = 1
            if(elegidos[ind[1]] == 0):
                poblOrdenada.append(self.pobl[ind[1]])
                elegidos[ind[1]] = 1
                
            maximo = np.amax(co)
            
        self.pobl = np.array(poblOrdenada)
        
class Cluster(object):  
    def __init__(self, arrayPos, centroide, variance, inputs):
        self.arrayPos = arrayPos
        self.centroide = centroide
        self.variance = variance
        self.numInputs = len(inputs)
        self.inputs = inputs
        
def neighbour(cl1, cl2, dist):
    for i in cl1.arrayPos:
        for j in cl2.arrayPos:
            i1 = np.array(i)
            j1 = np.array(j)
            if(np.sum(np.absolute(i1-j1))==dist):
                return True
    return False

def constraintsList(mat):
    devuelve = []
    longigrande = len(mat)
    longipeque = longigrande-1
    for i in range(0, longipeque):                        #De la ultima fila solo nos interesaria que el ultimo valor debe hacer link consigo
        for j in range(i+1, longigrande):                 #PARA QUE NO SE CUENTEN POR DUPLICADO NI LAS RESTRICCIONES DE UN VALOR CONSIGO MISMO
            if (mat[i][j] == 1.0):
                devuelve.append([i, j, 1.0])
            if (mat[i][j] == -1.0):
                devuelve.append([i, j, -1.0])
    return devuelve
