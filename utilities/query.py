# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from pprint import pprint
from urllib.parse import urljoin
import pandas as pd
import fileinput
import logging
import fasttext
import nltk
import re


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')
stemmer = nltk.stem.PorterStemmer()

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query

def transform_query(query: str, normalize: bool = False, stem: bool = False):
    if normalize:
        # Remove all non-alphanumeric characters other than underscore
        query = re.sub(r'[^\w_ ]', ' ', query)
        # Trim excess space characters so that tokens are separated by a single space.
        query = re.sub(r'\s+', ' ', query)
        # Convert all letters to lowercase and remove surround whitespace
        query = query.lower().strip()
    # Stem
    if stem:
        query = ' '.join([stemmer.stem(word) for word in query.split()])
    return query

# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", size=10, source=None, use_synonyms=False):
    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [

                        ],
                        "should": [  #
                            {
                                "match": {
                                    f"{'name.synonyms' if use_synonyms else 'name'}": {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": ["name^10", 
                                               "name.hyphens^10",
                                               "shortDescription^5",
                                               "longDescription^5", 
                                               "department^0.5", 
                                               "sku", 
                                               "manufacturer", 
                                               "features",
                                               "categoryPath", 
                                               "name_synonyms"]

                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        }
    }
    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj



def search(client, user_query, index="bbuy_products", sort="_score", sortDir="desc", use_synonyms=False, category=False):
    #### W3: classify the query
    #### W3: create filters and boosts
    # Note: you may also want to modify the `create_query` method above
    query_filter = None
    if category:
        query_filter = {"match": {"categoryPathIds": category}}
        
    query_obj = create_query(user_query,
        click_prior_query=None,
        filters=query_filter,
        sort=sort,
        sortDir=sortDir,
        source=["name", "shortDescription"],
        use_synonyms=use_synonyms)
    logger.info(query_obj)
    response = client.search(query_obj, index=index)
    if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
        hits = response['hits']['hits']
        # print(json.dumps(response, indent=2))


if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument("--synonyms", action=argparse.BooleanOptionalAction,
                         default=False, help="Whether to query the product title or synonyms")
    general.add_argument("--category_filter", action=argparse.BooleanOptionalAction,
                         default=False, help="Whether to fitler results to one category")
    general.add_argument('--user',
                         help='The OpenSearch admin. If this is set, the program will prompt for password too. If not set, use default of admin/admin')
    general.add_argument("--model_path", default="/workspace/datasets/fasttext/bbuy_query_stemmed_cat_clf_improved_model_1k.bin",
                         help="The path to the model")
    general.add_argument("--normalize", action=argparse.BooleanOptionalAction,
                    help="Normalize the product names by stripping symbols, applying lowercase and stemming")
    general.add_argument("--stem", action=argparse.BooleanOptionalAction,
                    help="Apply the Snowball stemmer to the queries")

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    host = args.host
    port = args.port
    model_path = args.model_path
    if args.user:
        password = getpass()
        auth = (args.user, password)

    use_synonyms = args.synonyms

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    index_name = args.index

    model = fasttext.load_model(model_path)


    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"

    while True:
        try:
            query: str = str(input(query_prompt)).rstrip()
        except KeyboardInterrupt:
            break
        else:
            if query.lower() == "exit":
                break
            else:
                top_cat, score = model.predict(transform_query(query, args.normalize, args.stem))
                if score[0] < 0.4 or not args.category_filter:
                    top_cat = False
                else:
                    print(top_cat, score)
                    top_cat = top_cat[0].replace('__label__', '')

                search(
                    client=opensearch,
                    user_query=query,
                    index=index_name,
                    use_synonyms=use_synonyms,
                    category=top_cat
                )

    # print(query_prompt)
    # for line in fileinput.input():
    #     query = line.rstrip()
    #     if query == "Exit":
    #         break
    #     search(client=opensearch, user_query=query, index=index_name, use_synonyms=use_synonyms)

    #     print(query_prompt)
