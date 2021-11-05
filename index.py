from elasticsearch import Elasticsearch,helpers
import sys,json,os



def delete_index(es, INDEX_NAME):
 if es.indices.exists(INDEX_NAME):
    print("deleting '%s' index..." % (INDEX_NAME))
    res = es.indices.delete(index = INDEX_NAME)
    print(" response: '%s'" % (res))

def load_json(directory):
    bulk_Data={}
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, directory)
    for filename in os.listdir(abs_file_path):
        if filename.endswith('.json'):
            open_file=open(abs_file_path+filename, 'r',encoding="utf8")
            js=json.load(open_file)
            if js["type"] == "blog":
                newJS = {"blog":
                           {
                                "url":js["blog_url"],
                               "title":js["blog_name"],
                               "posts":[]
                           }}
                for i in range(1,6):
                    if "post_url_"+str(i) in js:
                        newJS["blog"]["posts"].append(
                            {"post_url": js["post_url_"+str(i)],
                             "post_title": js["post_title_"+str(i)],
                            "post_content": js["post_content_"+str(i)],
                            "post_comments":[]
                             }
                        )
                    else: break;
                bulk_Data[js["blog_url"]]=newJS
            else:
                for item in bulk_Data.keys():
                    if str(js["blog_url"]) in item: #str ro bardaram check oknam
                        blog=bulk_Data[item]
                        for i in range(5):
                            if blog["blog"]["posts"][i]["post_url"]== js["post_url"]:
                                for it in js["comment_urls"]:
                                    blog["blog"]["posts"][i]["post_comments"].append({"comment_url":it})
                                break
                        break
    return bulk_Data


def index(adr):
    es = Elasticsearch(adr)
    print("Connected", es.info())
    global INDEX_NAME;
    INDEX_NAME="blog-index"
    delete_index(es,INDEX_NAME)
    es.indices.create(index=INDEX_NAME, ignore=400)
    data=list(load_json("hanie_output/").values())
    for i in range(len(data)):
        data[i]={"blog":data[i]["blog"]}
    helpers.bulk(es, data, index=INDEX_NAME, doc_type="blog")
    es.indices.refresh(index="blog-index")
    print("indexing finished")
    return es;


