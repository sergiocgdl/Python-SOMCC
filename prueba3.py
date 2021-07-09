import numpy as np
import matplotlib.pyplot as plt
import math
from som3 import CopSoft_SOM
from sklearn.metrics.cluster import adjusted_rand_score
import sys

################################################################################
#                Función para Lectura  y tratamiento de los datos
################################################################################
# Coge los datos, en el formato de los archivos de ejemplo y los pasa a forma matricial
# de forma que cada fila del archivo pasará a ser una fila en la matriz, y los valores
# por columnas en la fila son los especificados en la línea del archivo separados por columnas

def leerArchivo(nombrearchivo):
    archdatos = open(nombrearchivo, "r");
    datosini = archdatos.read()
    archdatos.close()

    #Obtengo cada ejemplo
    datosini = datosini.split("\r")         #Mítico retorno de carro que está oculto
    datosini = datosini[0].split("\n")      #Ya tengo cada ejemplo bien dividido

    if datosini[len(datosini)-1] == "":     #Para evitar algunas líneas vacías en ficheros de restricciones
        datosini.pop()


    for i in range(0, len(datosini)):    #Parto cada ejemplo en sus componentes
        datosini[i] = datosini[i].split(",")
        for j in range (0, len(datosini[i])):              #Reconvierto tipo de string a flotante
            datosini[i][j] = float(datosini[i][j])

    datosini = np.array(datosini)
    
    return datosini

def load_datasets(names, folder):
	names = np.sort(names)
	datasets_array = []
	labels_array = []
	names_array = []

	for i in range(len(names)):
		data = np.loadtxt(folder + "/" + names[i] + ".dat", delimiter = ",", dtype=str, comments = "@")
		data_set = np.asarray(data[:, :-1].astype(float))
		data_labels = np.asarray(data[:, -1])
		datasets_array.append(data_set)
		labels_array.append(data_labels)
		names_array.append(names[i])

	return names_array, datasets_array, labels_array
################################################################################
#                Función para construir lista de restricciones
################################################################################
# Coge una matriz de restricciones y la convierte en una lista de restricciones

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

sets = ["iris", "appendicitis", "balance", "ionosphere", "glass",
        "banana_undersmpl", "breast_cancer", "contraceptive", "ecoli", "haberman",
        "hayes_roth", "heart", "monk2", "newthyroid", 
        "page_blocks_undersmpl", "phoneme_undersmpl", "pima", "saheart", "satimage_undersmpl",
        "segment_undersmpl", "sonar", "soybean", "spambase_undersmpl", "spectfheart",
        "tae", "thyroid_undersmpl", "titanic_undersmpl", "vehicle", "wdbc",
        "wine", "zoo"]
        
cl = [3, 2, 3, 2, 7,
      2, 2, 3, 8, 2,
      3, 2, 6, 3,
      5, 2, 2, 2, 7,
      7, 2, 4, 2, 2,
      3, 3, 2, 4, 2,
      3, 7]

total = 0.0
ej = np.zeros(len(sets))
for i in range(len(sets)):
    x = np.loadtxt("./data/Datasets/Reales/"+ sets[i] +".dat", delimiter = ",", dtype=str, comments = "@")
    data = np.asarray(x[:, :-1].astype(float))
    labels = np.asarray(x[:, -1])
    restr = np.loadtxt("./data/Constraints/Reales/"+sets[i]+"("+sys.argv[1]+").txt", dtype=np.int8)
    listaRestr = constraintsList(restr)

    if(int(sys.argv[2])==2):
        shape = (3,3)
    else:
        shape = (2,2,2)
    ep = int(sys.argv[3])
    k = cl[i]
    
    media = 0.0
    ejec = np.zeros(10)
    for j in range(10): 
        som = CopSoft_SOM(shape, seed = j)  
        som.initialize(data, restr)
        som.fit(data, restr, ep, save_e=True, decay="")  # fit the SOM for 10000 epochs, save the error every 100 steps
        som.generateKClusters(data, restr, k)
        ejec[j] = adjusted_rand_score(labels, som.dataInCluster)
        media += ejec[j] 
        print("{0}, {1}".format(j, ejec[j]))
    media /= 10
    total += media
    
    desv = 0.0
    for j in range(10):
        desv += (ejec[j]-media)*(ejec[j]-media)
    desv /= 10
    ej[i] = media
    print("{0}, {1}, {2}, {3}, {4}".format(ep, shape, i, media, np.sqrt(desv)))

total /= len(sets)
des = 0.0
for i in range(len(sets)):
    des += (ej[i]-total)*(ej[i]-total)
des /= len(sets)

print("{0}, {1}, {2}, {3}".format(ep, shape, total, np.sqrt(des)))
