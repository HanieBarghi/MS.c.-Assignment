from scipy.sparse import csc_matrix
from index import *
from bidict import *
import numpy as np

def init():
    es=index("http://localhost:9200/")
    return es

def page_rank(G, maxerr=.0001):
    n = G.shape[0]

    M = csc_matrix(G, dtype=np.float)
    rsums = np.array(M.sum(1))[:, 0]
    ri, ci = M.nonzero()
    M.data /= rsums[ri]

    sink = rsums == 0

    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r - ro)) > maxerr:
        ro = r.copy()
        for i in range(0, n):
            Ii = np.array(M[:, i].todense())[:, 0]
            Si = sink / float(n)
            r[i] = ro.dot(Ii + Si)

    return r / sum(r)

def calc_page_rank():
    es=init()
    doc = {
        'size' : 10000,
        'query': {
            'match_all' : {}
       }
    }
    page = es.search(index="blog-index", doc_type='blog', body=doc,scroll='1m')
    scroll_size = page['hits']['total']
    print("scroll size: " + str(scroll_size))
    id_to_url=bidict()
    id_to_num=bidict()
    i=0
    for item in page['hits']['hits']:
        id_to_num[item['_id']]=i
        i=i+1
        id_to_url[item['_id']]=item['_source']['blog']['url']
    adj=np.zeros((scroll_size,scroll_size))
    for item in page['hits']['hits']:
        this_id=item['_id']
        for itemak in item['_source']['blog']['posts']:
            for urls in itemak["post_comments"]:
                res=urls["comment_url"]
                res=str(res)+"/"
                if res in id_to_url.inv:
                    adj[id_to_num[id_to_url.inv[res]],id_to_num[this_id]]+=1
    sums=adj.sum(axis=1)
    for j in range (len(sums)):
        if(sums[j]>0):
            adj[j, :] /= sums[j]
    alpha=0.01
    constant=np.inner(np.ones(len(sums)), alpha)
    adj=np.inner(adj, 1-alpha)
    adj=adj+constant

    pageranks=page_rank(adj)
    actions=[]
    for ind in range(len(pageranks)):
        q = {
            "_index": "blog-index",
            "_type": "blog",
            '_op_type': 'update',
            "_id": id_to_num.inv[ind],
            "script": {
                "inline": "ctx._source.blog.page_rank=params.rank",
                "params": {
                    "rank": pageranks[ind]
                }
            }
        }
        actions.append(q)

    helpers.bulk(es, actions, index='blog-index', doc_type="blog")
    print("page_rank updated successfully")
    return es
#calc_page_rank()
