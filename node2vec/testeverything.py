import networkx as nx
import pandas as pd
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from collections import defaultdict
from EvalUtils import EvalUtils

#g = nx.read_weighted_edgelist("../datasets/redditdataset_75.txt", create_using=nx.DiGraph())

df = pd.read_csv('../datasets/redditdataset.txt', names = ['v1','v2','timestamp'],sep = '\t',lineterminator='\n',header = None)

for i in range(1,5):
    randomrow = df.sample(1)
    randindex = randomrow.index.values.astype(int)[0]+1
    dftrain = df[:randindex]
    dftest =  df[randindex + 1:]

    predfor = str(df.at[randindex,'v1'])
    print("Predicting for:" + (predfor))
    actual = list(dftest[dftest["v1"].str.contains(predfor)]["v2"])
    if (len(actual) == 0 ):
        continue

    graph = nx.from_pandas_edgelist(dftrain,source='v1',
                                       target='v2',edge_attr='timestamp',
                                       create_using=nx.DiGraph())
    node2vec = Node2Vec(graph, num_walks=10, walk_length=8)


    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)


    listobj = model.wv.most_similar(predfor)[:10]
    nodes = [elem[0] for elem in listobj]
    pred_edges = [(predfor, node) for node in nodes]
    pred_set = set(pred_edges)

    preds = len(set(actual).intersection(pred_set))

    print("Round" + str(i) + "\n")
    print(EvalUtils.mapk(actual,list(pred_set),k=10))
    print(preds/len(actual))
    print("\n")