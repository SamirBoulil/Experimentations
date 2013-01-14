#!/usr/bin/env python
# coding=utf-8

import Orange
import orngCA
import sys, os
import csv
import math
import shutil
from matplotlib import pyplot
# from pygooglechart import Chart
# from pygooglechart import SimpleLineChart
# from pygooglechart import Axis

EPSILON = 0.00000000000001


DISTANCE_FILES = 'DISTANCE_FILES'
DISTANCE_FILE_NAME=0
OPTIMUM_NB_CLUSTER_INDEX=1
RESULTS_PATH_INDEX=2
OPTIMUM_NB_CLUSTER_VALUE=3

NB_CLUSTER_REF = 'NB_CLUSTER_REF'
NB_DISTANCE_FILE='NB_DISTANCE_FILE'
FILENAME = "Precision_Rappel"
SOLUTIONPATH = "resultats/"
PATH_EXPERIMENTS = "experimentations/"

####################################################################################
#Ecriture disque

#Ecriture d'un rapport récapitulant les 
def writeSummaryResults(filename, summaryOfProcess):
    with open(filename, 'wb') as summary:
        summary.write("# RESULTS OF EXPERIMENTATIONS :\n")
        summary.write("## General information :\n")
        summary.write("\t Number of cluster reference : %s \n" % summaryOfProcess[NB_CLUSTER_REF])
        summary.write("\t Number of distance file treated : %s \n" % summaryOfProcess[NB_DISTANCE_FILE])
        for val in summaryOfProcess[DISTANCE_FILES]:
            summary.write("## Results for distance file name : %s \n" % val[DISTANCE_FILE_NAME])
            summary.write("\t - Result path is : \"%s\" \n" % val[RESULTS_PATH_INDEX])
            summary.write("\t - Optimum cluster for fmesure at index : %s for FMesure = %s \n" % (val[OPTIMUM_NB_CLUSTER_INDEX], val[OPTIMUM_NB_CLUSTER_VALUE]))
        summary.write("You can find all the best results in the "+SOLUTIONPATH+" directory.")
        
            
        

#Ecriture Des résultats pour une comparaison entre un groupement de la CAH et les clusters de références avec toutes les mesures
def writeResults(filename, resMatrix, dataTuple, precisionAVG, recallAVG, FMesureAVG, ecartTypeAVGFMesure, clusterRef, clusterPredit):
    result_matrix = [];
    header = [];
    #Pour chaque colonne j'ai mon cluster expérimenté
    for i in range(len(resMatrix[0])+1):
        if i>0:
            header.append("Cluster "+str(i-1))
        else:
            header.append("(PRECISION/RAPPEL/FMESURE)")
    result_matrix.append(header);
    
    #Pour chaque ligne j'ai un label de cluster reference
    for i,item1 in enumerate(resMatrix):
        temp = []
        temp.append("Reference cluster "+str(i))
        for j,item2 in enumerate(item1):
            temp.append(item2)
        result_matrix.append(temp)
    
    #Header pour les détails des clusters Reference
    tempClusterRef = []
    for i, row in enumerate(clusterRef):
        tempRow = []
        tempRow.append("Reference cluster "+str(i))
        for item in row:
            tempRow.append(item)
        tempClusterRef.append(tempRow)
        
    
    
    
    #Correspondance entre REF <-> predit
    tempCorrespondate = []
    for i, corr in enumerate(dataTuple):
        # -> Cluster "+stockéer(dataTuple[i][3])+" avec FMesure = "+str(dataTuple[i][2])+"."
        tempRow = []
        tempRow.append("Reference clusters "+str(i))
        tempRow.append("Cluster "+str(corr[3]))
        tempRow.append((corr[0],corr[1],corr[2]))#PRECISION/RAPPEL/FMEsure
        tempRow.append(tempClusterRef[i])#ajout de la liste de mots du cluster référence
        tempRow.append(clusterPredit[corr[3]])#ajout de la liste de mots du cluster prédit
        tempCorrespondate.append(tempRow)
    
    #Header pour les détails des clusters prédits
    for i, row in enumerate(clusterPredit):
        row.insert(0, "Cluster "+str(i))

    #Ecriture dans le fichier de la matrice
    with open(filename, 'wb') as test_file:
        file_writer = csv.writer(test_file)
        file_writer.writerow(["MATRICE DE COMPARAISON", "Precision AVG (%) "+str(precisionAVG), "Recall AVG (%)"+str(recallAVG),"FMesure AVG (%): "+str(FMesureAVG), "FMesure ecart-type : "+str(ecartTypeAVGFMesure)])
        file_writer.writerows(result_matrix)
        file_writer.writerow([""])
        file_writer.writerow(["CORRESPONDANCES"])
        file_writer.writerow(["REFERENCE", "CLUSTER", "PRECISON/RAPPEL/FMESURE"])
        for i, corr in enumerate(tempCorrespondate):
            file_writer.writerow(corr[0:3])#On écrit d'abord la correspondances avec les mesures
            file_writer.writerow(corr[3])#Puis on donne la liste des termes pour le référence
            file_writer.writerow(corr[4])#Puis la liste des termes pour le détail
            file_writer.writerow(["_____________________"]*max(len(corr[3]),len(corr[4])))#Séparation de la taille de la plus grande des deux listes de termes
        file_writer.writerow([""])
        file_writer.writerow(["CLUSTER DE REFERENCE"])
        file_writer.writerows(tempClusterRef)
        file_writer.writerow([""])
        file_writer.writerow(["CLUSTER PRÉDITS"])
        file_writer.writerows(clusterPredit)

def loadCSVasListOfCLusters(path):
    clusters = [];
    i=0;
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            clusters.append(row);
            i = i+len(row);
    return  clusters, i;

# Fonction qui charge en mémoire une matrice de distance stockée dans un fichier
# paramètre : matrixPath - Chemin vers le fichier
# retourne : les labels correspondant aux indices de la matrice de distance, la matrice de distancechargée dans un objet Orange
def loadDistanceMatrix(matrixPath):
    #print "Loading distance matrix"
    i=0
    j=0
    label = []
    buf = []
    distance = []
    
    with open(matrixPath, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            if i > 0:
                j=0
                buf = []
                for item in row:
                    if j == 0:
                        label.append(item);
                    else:
                        buf.append(item);
                    j += 1
                distance.append(buf)
            i += 1;

    #On remplit les valeurs dans une matrice symMatrix utilisés par orange pour le clustering
    orangeDistance = Orange.misc.SymMatrix(len(distance));
    for i in range(len(distance)):
        row = distance[i]
        for j in range(len(row)):
            orangeDistance[i,j] = row[j]
    return label, orangeDistance

####################################################################################
#Calculs le score de précision de deux groupes
# paramètre : clusterReference, clusterPredit : des listes de mots
# retourne une double représentant la precision (IR)
def precision(clusterReference, clusterPredit):
    tp=0.0;
    fp=0.0;
    
    for item1 in clusterPredit:
        if item1 in clusterReference:
            tp+=1.0;
        else:
            fp+=1.0;
    
    precision = (tp/(tp+fp))
    return precision

#Calculs le score de rappel de deux groupes
# paramètre : clusterReference, clusterPredit : des listes de mots
# retourne une double représentant le rappel (IR)
def rappel(clusterReference, clusterPredit):
    tp=0.0;
    fn=0.0;
    
    for item1 in clusterReference:
        if item1 in clusterPredit:
            tp+=1.0;
        else:
            fn+=1.0;
    
    recall = (tp/(tp+fn))
    return recall;


#Calculs la fMesure de deux groupes
# paramètre : clusterReference, clusterPredit : des listes de mots
# retourne une double représentant la fmesure (IR)
def fMesure(clusterReference, clusterPredit):
    precis = precision(clusterReference,clusterPredit)
    rapp = rappel(clusterReference, clusterPredit)
    
    return 2.0*((precis*rapp)/(precis+rapp+EPSILON))    
####################################################################################
#
#Fonction qui effectue un CAH à partir d'une matrice de distance
# Paramètre : distanceMatrix : L'objet orange matrice de distance
# Retourne : un objet orange, représentant le résultat de la CAH 
def clusterDistances(distanceMatrix):
    #print "Clustering the distance matrix"
    clustering = Orange.clustering.hierarchical.HierarchicalClustering()
    clustering.linkage = Orange.clustering.hierarchical.WARD
    return clustering(distanceMatrix)

#Fonction qui représente sous forme de dendogramme les 3 premiers clusters
# Paramètres : root - le noeud racine de l'arbre de la CAH
def showTroisGroupes(root):
    topmost = sorted(Orange.clustering.hierarchical.top_clusters(root, 3), key=len)

    for n, cluster in enumerate(topmost):
        print "\n\n Cluster %i \n" % n
        for instance in cluster:
            print labels[instance]
    print
    
    my_colors = [(255,0,0), (0,255,0), (0,0,255)]
    colors = dict([(cl, col) for cl, col in zip(topmost, my_colors)])
    
    Orange.clustering.hierarchical.dendrogram_draw("hclust-dendrogram.png", root, labels=labels, cluster_colors=colors, color_palette=[(0,255,0), (0,0,0), (255,0,0)],
    gamma=0.5, minv=2.0, maxv=7.0)


# Fonction qui permet de générer un tableau CSV de précisions/rappels/FMesure pour chaque cluster de référence et prédit passés en paramètre. et retourne ce tableau
# En effet, le résultat est alors sous la forme :
# |     | predit 1              | predit ... | predit n
# |ref 1|(pres1,rap1,fmesure1)  |  ...       | ...
# | ... |
# |ref n|
def compareClusters(clusterReference, clusterPredit):
    i = 0;
    j = 0;
    result_matrix = [[0 for x in xrange(len(clusterPredit))] for x in xrange(len(clusterReference))];#En ligne les clusters référent, en colonne les cluster prédit de la CAH
    
    for i in range(len(clusterReference)):
        for j in range(len(clusterPredit)):
            currPrec = precision(clusterReference[i], clusterPredit[j])
            currRapp = rappel(clusterReference[i], clusterPredit[j])
            currFMes = fMesure(clusterReference[i], clusterPredit[j])
            result_matrix[i][j] = (currPrec, currRapp, currFMes)
    #print "** Ecriture du fichier :"+str(filename)
    return result_matrix


def printHierarchicalCluster(hc,labels):
    for n, cluster in enumerate(hc):
        print "\n\n Cluster %i \n" % n
        for instance in cluster:
            print instance
    print

def clustersIdToClustersLabel(cluster, labels):
    res = [];
    for n, cluster in enumerate(cluster):
        res.append([])
        for instance in cluster:
            res[n].append(labels[instance])
    return res;
#####################################################################################
#fonction qui retourne les valeurs ayant le plus de sens en fonction des clusters Prédit et de référence (eg, en choisissant la meilleur case pour les clusters. 
# Pour chaque cluster de référence, chercher le cluster prédit qui maximise la mesure en question et l'éliminer de la liste des clusters disponibles
# Dans l'idéal si on trouve qu'un cluster déjà éliminé maximise encore un cluster de référence. On ira vérifier si la valeur est alors plus grande et on switchera alors
# Ca m'a lair compliqué.
#Retourne un liste de toute les valeurs sélectionnées par l'algorithme.
# Où measureIndex=0 : le tuple, en maximisant la precision
# Où measureIndex=1 : le tuple, en maximisant la rappel
# Où measureIndex=2 : le tuple, en maximisant la FMesure
def getRelevantData(result_matrix, measureIndex):
    correspondance = [None]*len(result_matrix)#Taille = nombre de cluster de référence    []#[(precision, rappe, FMesure, CLUSTER)]
    unavailablePredictedCluster = []# On met les indices des clusters prédit qui ont été affectés ?
    problematic_Ref_clusters = []# On met ici les indices des clusters de référence dont on ne peut définir directement les bons clusters
    
    #Pour faire des calculs plus simple on récupère les valeurs de la mesure dont on souhaite
    # result_matrix_measure = []
    # for reference in result_matrix:
    #     temp = []
    #     for predit in reference:
    #         temp.append(predit[measureIndex])
    #     result_matrix_measure.append(temp)
    # print result_matrix_measure
    
    ############################
    #Premier tour : Les équivalence sans équivoque REFERENCE -> PREDIT
    # tempRes = []
    #Où tempRes :
    # [#reference1
    #         [
    #             (reference1 -> predit 2),
    #             (reference2 -> predit 4),
    #             
    #         ]
    #         
    #     ]
    for i,reference in enumerate(result_matrix):
        maxVal = max(reference, key=lambda x: x[measureIndex])[measureIndex]#récupération du max sur la ligne
        indexesLigne = []
        #Vérification pour reference -> ONE the_predit
        for j,predit in enumerate(reference):
            if predit[measureIndex] >= maxVal:
                maxVal = predit[measureIndex]
                indexesLigne.append((i, j, maxVal))

        #Si cluster non problématique : reference -> ONE predit      
        if len(indexesLigne) == 1:
            maxVal = indexesLigne[0][measureIndex]#FMesure max en ligne
            preditIndex = indexesLigne[0][1]
            indexesColonne = []
            for j in range(len(result_matrix)):
                if result_matrix[j][preditIndex][measureIndex] >= maxVal:
                    indexesColonne.append((j,preditIndex))
        
        if len(indexesLigne) == 1 and len(indexesColonne) == 1 :
            #On a une correspondance
            corr_tuple = result_matrix[i][preditIndex]+(preditIndex,)
            correspondance[i] = corr_tuple

    ############################
    #deuxième tour : Les équivalence compliquées
    #Vérification pour !AutreReference -> the_Predit
    for x in correspondance:#récupération des indexes de clusters prédit déjà pris
        if x is not None:
            unavailablePredictedCluster.append(x[3])
    # print correspondance
    # print unavailablePredictedCluster
    
    for indexNone, corr in enumerate(correspondance):
        if corr is None:
            # print "treating REF None : "+str(indexNone)
            #determine le max pour le référent à indexNone
            reference = result_matrix[indexNone]
            #Recherche du max
            maxVal = -1
            indexMax = None
            # print reference
            for j, predit in enumerate(reference):
                if predit[measureIndex] >= maxVal  and j not in unavailablePredictedCluster:
                    maxVal = predit[measureIndex]
                    indexMax = j
                    #print str(j)+"- "+str(maxVal)
                        
            #Affectation
            if indexMax is None :
                print "Pas de correspondance REF "+str(i)
                
            elif indexMax not in unavailablePredictedCluster:
                    
                    #print "for reference "+str(indexNone)+" Found correspondance with cluster : "+str(indexMax)+" for FMesure "+str(maxVal)
                    correspondance[indexNone] = result_matrix[indexNone][indexMax]+(indexMax,)#On choisit le max et on l'ajoute
                    unavailablePredictedCluster.append(indexMax)
                    
                    if result_matrix[indexNone][indexMax][measureIndex] != maxVal:
                        print str(result_matrix[i][indexMax]+(indexMax,))+" - "+str(maxVal)
                        print "Top at index "+str(indexMax)+" For reference : "+str(i)
                        sys.exit("bite")
                    
    # print unavailablePredictedCluster
    # print correspondance
    
    return correspondance
    
        
# Fonction qui calcule la moyenne de mesure sur notre matrice.
# Ou measureIndex=0 : precision
# Ou measureIndex=1 : rappel
# Ou measureIndex=2 : FMesure
def getMeasureAVG(dataTuples, measureIndex):
    data = [x[measureIndex] for x in dataTuples]
    return float(sum(data)/len(data))    

# http://stephane.bunel.org/Divers/moyenne-ecart-type-mediane
def getMeasureEcartType(dataTuples, measureIndex):
    dataAVG = getMeasureAVG(dataTuples, measureIndex)**2
    somme = sum([x[measureIndex]**2 for x in dataTuples])
    variance = float(somme/len(dataTuples) - dataAVG)
    return float(math.sqrt(variance))
     


def generateGraph(fMesureGlobalPerCent, bottom_limit, nbComparison, resPath, title, optimumSolution):
    
    X = range(bottom_limit,nbComparison+1)
    Y = fMesureGlobalPerCent
    pyplot.plot( X, Y, ':rs' )
    pyplot.ylim([0,109])
    pyplot.xlim([0,nbComparison+1])
    pyplot.title(title)
    pyplot.xlabel( 'Number of clusters' )
    pyplot.ylabel( '%' )
    if optimumSolution is not None:
        pyplot.annotate("MAX "+str(optimumSolution), xy=optimumSolution, xytext=(20,100),arrowprops=dict(facecolor='black', shrink=0.05),)
    pyplot.savefig(resPath)
    pyplot.close()
    # pyplot.show() #Interactive
        
    
def moveToBestSolutionPath(resPath, distanceFile, optimumSolution):
    filename_old = resPath+FILENAME+str(optimumSolution[0])+".csv"
    filename_new = SOLUTIONPATH+distanceFile+"_"+FILENAME+str(optimumSolution[0])+".csv"
    if not os.path.exists(SOLUTIONPATH):
        os.makedirs(SOLUTIONPATH)
        
    shutil.copy2(filename_old, filename_new)
    shutil.copy2(resPath+"FMesure-evolution_"+distanceFile+".png", SOLUTIONPATH+"FMesure-evolution_"+distanceFile+".png")
    
    

def main():
    summaryOfProcess={}
    print
    print "** LOADING REFERENCE CLUSTERS"
    clusterReference, i = loadCSVasListOfCLusters(sys.argv[1])
    numberOfReference = len(clusterReference)
    print"Reference : Loaded "+str(numberOfReference)+" clusters and "+str(i)+" concepts."
    print
    distancesFile = sys.argv[2::]
    summaryOfProcess[NB_DISTANCE_FILE]=str(len(distancesFile))
    summaryOfProcess[NB_CLUSTER_REF]=str(numberOfReference)
    summaryOfProcess[DISTANCE_FILES] = []
    
    for i, df in enumerate(distancesFile):
        print "** PROCESSING  DISTANCE FILE : "+df
        #Create result directory if not exists
        distanceFile = os.path.splitext(os.path.basename(df))[0]
        resPath = str(i)+"_"+distanceFile+"_"+PATH_EXPERIMENTS #Dossier du résultat de la forme <iteration>_<fichierDistance>_resultats/
        if not os.path.exists(resPath):
            os.makedirs(resPath)
            
        labels, matrix_distance = loadDistanceMatrix(df)
        root = clusterDistances(matrix_distance)
        
        #Pour chaque groupement possible faire topmost
        fMesureGlobalPerCent = []
        precisonGlobalPerCent = []
        recallGlobalPerCent = []
        ecartTypeGlobalPer = []
        for i in range(numberOfReference, len(matrix_distance[0])):
            filename = resPath+FILENAME+str(i)+".csv"
            clustersPredit = sorted(Orange.clustering.hierarchical.top_clusters(root,i), key=len) #CAH
            clustersPredit = clustersIdToClustersLabel(clustersPredit,labels) #On remplace les ID de mots par leurs noms
            result_matrix = compareClusters(clusterReference, clustersPredit) #On compare les clusters en calculant la précison/Rappel/FMesure
            
            dataTuple = getRelevantData(result_matrix, 2)#Recherche des correspondantes ref/prédit qui maximise la FMesure
            precisionAVG = float(getMeasureAVG(dataTuple, 0)*100)
            recallAVG = float(getMeasureAVG(dataTuple, 1)*100)
            FMesureAVG = float(getMeasureAVG(dataTuple, 2)*100) #On calcule la FMesure globale
            ecartTypeFMesureAVG = float(getMeasureEcartType(dataTuple,2)*100)
            
            precisonGlobalPerCent.append(precisionAVG) #On calcule la precision global et on ajoute dans la liste
            recallGlobalPerCent.append(recallAVG) #On calcule le rappel global et on ajoute dans la liste
            fMesureGlobalPerCent.append(FMesureAVG) #On sauve cette FMesure pour générer un graphe de l'évolution de la FMesure
            ecartTypeGlobalPer.append(ecartTypeFMesureAVG)
            writeResults(filename, result_matrix, dataTuple, precisionAVG, recallAVG, FMesureAVG, ecartTypeFMesureAVG,clusterReference, clustersPredit) #On écrit le résultat dans le fichier
        
        #Résumé du traitement
        optimumSolutionFM = (fMesureGlobalPerCent.index(max(fMesureGlobalPerCent))+numberOfReference, max(fMesureGlobalPerCent))
        print "maximum is at cluster :"+str(optimumSolutionFM[0])+" for fmesure = "+str(optimumSolutionFM[1])
        print "Results for "+df+" written at "+resPath
        print
        summaryTemp = [None]*4
        summaryTemp[DISTANCE_FILE_NAME]= df
        summaryTemp[OPTIMUM_NB_CLUSTER_INDEX] = str(optimumSolutionFM[0])#normal
        summaryTemp[OPTIMUM_NB_CLUSTER_VALUE] = str(optimumSolutionFM[1])
        summaryTemp[RESULTS_PATH_INDEX] = resPath
        summaryOfProcess[DISTANCE_FILES].append(summaryTemp)
        
        #Génération d'un graphe récapitulatif de la mesure
        generateGraph(fMesureGlobalPerCent, numberOfReference, i,resPath+"FMesure-evolution_"+distanceFile+".png", "Evolution de la FMesure", optimumSolutionFM)
        optimumSolution = (precisonGlobalPerCent.index(max(precisonGlobalPerCent))+numberOfReference, max(precisonGlobalPerCent))
        generateGraph(precisonGlobalPerCent, numberOfReference, i, resPath+"Precision-evolution_"+distanceFile+".png", "Evolution de la precision", optimumSolution)
        optimumSolution = (recallGlobalPerCent.index(max(recallGlobalPerCent))+numberOfReference, max(recallGlobalPerCent))
        generateGraph(recallGlobalPerCent, numberOfReference, i, resPath+"Recall-evolution_"+distanceFile+".png", "Evolution du rappel", optimumSolution)
        optimumSolution = (ecartTypeGlobalPer.index(max(ecartTypeGlobalPer))+numberOfReference, max(ecartTypeGlobalPer))
        generateGraph(ecartTypeGlobalPer, numberOfReference, i, resPath+"ecartType-Fmesure-evolution_"+distanceFile+".png", "Evolution de l'ecart type", optimumSolution)
        
        #Moving best solutions to solution_PATH
        moveToBestSolutionPath(resPath, distanceFile, optimumSolutionFM)
        
    #Génerating summary of process report file
    writeSummaryResults("SUMMARY.md",summaryOfProcess)
        
####################################################################################
#Tests
def tests():
    listA = ["a", "b", "c"]
    listA2 = ["a", "c"]
    listB = ["a", "b", "c", "d"]
    listC = ["e", "f", "g"]
        
    assert precision(listA, listA) == 1.0
    assert precision(listA, listA2) == 1.0
    assert precision(listA, listB) == 0.75
    assert precision(listA, listC) == 0.0
    
    
    assert rappel(listA, listB) == 1.0
    assert rappel(listA, listA2) == 0.6666666666666666


    b = [(0.25, 0.3333333333333333, 0.2857142857142808, 10), (0.0, 0.0, 0.0, 11), (0.0, 0.0, 0.0, 9), (1.0, 0.14285714285714285, 0.2499999999999978, 4), (0.0, 0.0, 0.0, 7), (0.10256410256410256, 0.6666666666666666, 0.17777777777777545, 13), (0.0, 0.0, 0.0, 5), (0.25, 0.5, 0.3333333333333289, 12), (0.0, 0.0, 0.0, 3), (0.0, 0.0, 0.0, 2), (0.0, 0.0, 0.0, 1), (0.3333333333333333, 0.16666666666666666, 0.22222222222221777, 8), (0.0, 0.0, 0.0, 0), (1.0, 0.2, 0.3333333333333306, 6)]
    assert getMeasureAVG(b,2)-0.114455782313 <= EPSILON

    print "Tests ok"
    data = [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.25, 0.3333333333333333, 0.2857142857142808), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.02564102564102564, 0.6666666666666666, 0.049382716049382)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.038461538461538464, 1.0, 0.07407407407407336)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.038461538461538464, 0.75, 0.07317073170731615)], [(1.0, 0.14285714285714285, 0.2499999999999978), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.14285714285714285, 0.2499999999999978), (1.0, 0.14285714285714285, 0.2499999999999978), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.25, 0.14285714285714285, 0.1818181818181772), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.038461538461538464, 0.42857142857142855, 0.07058823529411613)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.038461538461538464, 1.0, 0.07407407407407336)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.08333333333333333, 0.1428571428571404), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.16666666666666666, 0.08333333333333333, 0.11111111111110666), (0.0, 0.0, 0.0), (0.10256410256410256, 0.6666666666666666, 0.17777777777777545)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.125, 0.25, 0.16666666666666222), (0.038461538461538464, 0.75, 0.07317073170731615)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.16666666666666666, 0.25, 0.1999999999999952), (0.25, 0.5, 0.3333333333333289), (0.01282051282051282, 0.25, 0.024390243902438095)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.16666666666666666, 0.14285714285714285, 0.15384615384614886), (0.125, 0.14285714285714285, 0.13333333333332836), (0.05128205128205128, 0.5714285714285714, 0.094117647058822)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.038461538461538464, 0.5, 0.0714285714285701)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.07692307692307693, 0.8571428571428571, 0.1411764705882338)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.3333333333333333, 0.16666666666666666, 0.22222222222221777), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.05128205128205128, 0.6666666666666666, 0.0952380952380939)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.038461538461538464, 0.6, 0.0722891566265049)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.2, 0.3333333333333306), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.2, 0.28571428571428165), (0.0, 0.0, 0.0), (0.125, 0.1, 0.11111111111110618), (0.05128205128205128, 0.4, 0.0909090909090889)]]
    b = getRelevantData(data,2)#get FMesure
    
if __name__ == '__main__':
    if len(sys.argv) > 2:
        tests()
        main()
    else:
        sys.exit("too few arguments : usage python compare.py <refCluster1> <distanceFile1> <distanceFile2> ...")

# Idées :
#   - Ne comparer que coupes qui ont au moin un nombre = au nombre de clusters prédit
#   - Faire une meilleure recherche pour le calcul des FMesure globales
########################################
#Garbage 
    #print "** les TOPS"
    #   On récupère le meilleur candidats pour chaque clusters de références
    #    topsPrecision=[-1 for x in range(len(refClusters))]
    #    topsRecall=[-1 for x in range(len(refClusters))]
    #    topsFMesure = [-1 for x in range(len(refClusters))]
    #    maxPrecision = -1.0;
    #    maxRappel = -1.0;
    #    maxFMesure = -1.0
    #    i=0;
    #    
    #    for ref1 in result_matrix:
    #        j=0;
    #        maxPrecision = -1.0
    #        maxRappel = -1.0
    #        for clust in ref1:
    #            if clust[0] > maxPrecision:
    #                maxPrecision = clust[0]
    #                topsPrecision[i] = (j+1, maxPrecision)
    #            if clust[1] > maxRappel:
    #                maxRappel = clust[1]
    #                topsRecall[i] = (j+1, maxRappel)
    #            if clust[2] > maxFMesure:
    #                maxFMesure = clust[2]
    #                topsFMesure[i] = (j+1, maxFMesure)
    #            j+=1
    #        i+=1
    #
    #    print "Top precision for each cluster ("+str(len(topsPrecision))+") :"+str(topsPrecision)
    #    print "Top recall for each cluster :"+str(topsRecall)
    #    print "Top FMesure for each cluster:"+str(topsFMesure)
    
    
    #previous presented results
    # for item1 in result_matrix:
    #     maxVal = 0.0
    #     for i,item2 in enumerate(item1):
    #         if item2[2] > maxVal and i not in eliminatedClust:
    #             if measureIndex == -1:
    #                 res.append(item2)
    #                 maxVal = item2[2]
    #             else:
    #                 maxVal = float(item2[measureIndex])
    #                 res.append(item2[measureIndex])
    #                 eliminatedClust.append(i)
    #         #else si FMesure supérieure
    
    
    
    # chart = SimpleLineChart(500, 500, y_range=[0, max_y])
    # chart.add_data(fMesureGlobalPerCent)
    # chart.set_colours(['0000FF'])
    # chart.fill_linear_stripes(Chart.CHART, 0, 'CCCCCC', 0.2, 'FFFFFF', 0.2)
    # chart.set_grid(0, 25, 5, 5)
    # chart.add_data(max(fMesureGlobalPerCent))
    #    
    # left_axis = range(0, max_y + 1, 25)
    # left_axis[0] = ''
    # chart.set_axis_labels(Axis.LEFT, left_axis)
    # 
    # y_axis = []
    # for x in range(bottom_limit,nbComparison):
    #     if x%5 == 0:
    #         y_axis.append(x)
    # chart.set_axis_labels(Axis.BOTTOM, y_axis)
    # chart.download(resPath)
    
    
