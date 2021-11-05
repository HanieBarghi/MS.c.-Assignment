from pageRank import *

def ES_search(es):
    titleQuery="سلام"; contentQuery="من"; postQuery="با"
    #titleQuery=input("Enter blog title of your query: ")
    #postQuery= input("Enter Post Title Query: ")
    #contentQuery= input("Enter content Query: ")
    #titleScore=int(input("Enter title Score: "))
    #contentScore=int(input("Enter post content Score: "))
    #postScore=int(input("Enter post title Score: "))
    titleScore=5;    contentScore=1; postScore=1
    pageRank=input("Is page rank important in result? (Y/N) ")
    if(pageRank=="N"):
        res = es.search( body={"query": {
        "bool": {
            "should": [
                {"match": {
                    "blog.posts.post_title":
                    {
                        "query": postQuery,
                        "boost": postScore
                    }}},
                {"match":{
                    "blog.posts.post_content":
                        {
                            "query": contentQuery,
                            "boost": contentScore
                        }
                }},
                {"match": {
                    "blog.title":
                        {
                            "query": titleQuery,
                            "boost": titleScore
                        }}}
            ],
            "minimum_should_match": 2
        }
        }})
    else:
        res = es.search(body={"query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {
                                "blog.posts.post_title":
                                    {
                                        "query": postQuery,
                                        "boost": postScore
                                    }}},
                            {"match": {
                                "blog.posts.post_content":
                                    {
                                        "query": contentQuery,
                                        "boost": contentScore
                                    }
                            }},
                            {"match": {
                                "blog.title":
                                    {
                                        "query": titleQuery,
                                        "boost": titleScore
                                    }}}
                        ],
                        "minimum_should_match": 2
                    }},
                "script_score": {
                    "script": {
                        "source": "_score* doc['blog.page_rank'].value"
                    }
                }
            }
        }})
    print("Got %d Hits:" % res['hits']['total'])
    print(res['hits'])
    for item in res['hits']['hits']:
        print("id: "+item['_id'])
        print("page rank: "+str(item['_source']['blog']['page_rank']))
        print("url: "+item['_source']['blog']['url'])
        print("title: "+item['_source']['blog']['title'])
        print("posts: \n")
        for itemak in item['_source']['blog']['posts']:
            print("post title:" +itemak["post_title"])
            print("post content:" +itemak["post_content"])

es=calc_page_rank()
ES_search(es)
