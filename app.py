import re
import sys
from pprint import pprint

import inflect
import pandas as pd
import weaviate
from SPARQLWrapper import SPARQLWrapper, JSON
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAI
from retrying import retry
from weaviate.collections.classes.config import Configure, VectorDistances, Property, DataType
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5

app = Flask(__name__)
CORS(app)

sys.stdout.reconfigure(encoding='utf-8')

PALM_API_KEY = "ya29.a0AXooCgvrJXKONOU9MP0o8HQUOgQapipCT1KrevzzrDMbsHU5EyFUXdkiR1Hp4LyO_ju37ESmARECJOFc68DvTuuuhP5PP0kqOpZT5xwA-jVvQFr13fp7E8ce4keISZZcprR5GZHlVB1nmi8TQjTdMc8Z0jv_KH-2aS0wyqSSsIe_BocizwkC3NBsBdbEg7EjNWUbLIcVfupojhmkHYVdND8gSzQt4sA18eKOmX2_jL5C55A9uohhQmkntpgIuUhP54kpxh79p2OnhODbDIzWe8u8x3ufylkIm5M2mbEXNDS9h5pG1xSxWg_8pVpDXZ9tjBy5ffJepE7LCobaNZMy-drmZHlf9ZErXwc8mhcLUx-CrTwX5UCAjoVloGR1i5KuIuoE7R-Xkaka8Z9jQ5RmOU7XOVRm0gQSsDEaCgYKASkSARMSFQHGX2Miv8YdEwJUnW4Cwc-pcnUwIA0426"
GOOGLE_API_KEY = "AIzaSyAJ6mi9i3I5qnEGgwJql4eJc6CZULfcYKU"

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY)

client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    headers={
        "X-PaLM-Api-Key": PALM_API_KEY
    },
    skip_init_checks=True
)
print("Weaviate client connected")

url = 'https://raw.githubusercontent.com/IoanCosarca/Jeopardy-Dataset/main/jeopardy_questions.csv'
df = pd.read_csv(url, sep=',', nrows=1000)

client.collections.delete("JeopardyQuestion")
client.collections.delete("DBpediaArticle")

client.collections.create(
    name="JeopardyQuestion",
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE
    ),
    properties=[
        Property(name="category", data_type=DataType.TEXT),
        Property(name="question", data_type=DataType.TEXT),
        Property(name="answer", data_type=DataType.TEXT)
    ],
    vectorizer_config=Configure.Vectorizer.text2vec_palm(
        model_id="textembedding-gecko@001",
        vectorize_collection_name=False,
        project_id="t-monument-297120"
    ),
    generative_config=Configure.Generative.palm(
        temperature=0.25,
        project_id="t-monument-297120"
    ),
)

jeopardy = client.collections.get("JeopardyQuestion")

with client.batch.fixed_size(batch_size=100) as batch:
    for _, row in df.iterrows():
        question_object = {
            "category": row['category'],
            "question": row['question'],
            "answer": row['answer'],
        }
        batch.add_object(
            collection="JeopardyQuestion",
            properties=question_object,
            uuid=generate_uuid5(question_object)
        )
print("Added questions to DB")

client.collections.create(
    name="DBpediaArticle",
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE
    ),
    properties=[
        Property(name="abstract", data_type=DataType.TEXT)  # dbo:abstract
    ],
    vectorizer_config=Configure.Vectorizer.text2vec_palm(
        model_id="textembedding-gecko@001",
        vectorize_collection_name=False,
        project_id="t-monument-297120"
    ),
    generative_config=Configure.Generative.palm(
        temperature=0.25,
        project_id="t-monument-297120"
    )
)

dbpedia = client.collections.get("DBpediaArticle")

sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

print("Initializing sparql")
sparql_interrogation = f"""
    SELECT DISTINCT ?entity ?abstract ?url WHERE {{
        ?entity rdf:type dbo:MilitaryConflict ;
                dbo:date ?date ;
                dbo:abstract ?abstract ;
                foaf:isPrimaryTopicOf ?url .
        FILTER (?date >= "1900-01-01"^^xsd:date && ?date <= "1999-12-31"^^xsd:date)
        FILTER (CONTAINS(?abstract, 'Europe') = true)
        FILTER (lang(?abstract) = 'en')
    }}
"""
sparql.setQuery(sparql_interrogation)

try:
    with client.batch.fixed_size(batch_size=100) as b:
        initial_results = sparql.query().convert()
        if initial_results["results"]["bindings"]:
            for r in initial_results["results"]["bindings"]:
                a = r["abstract"]["value"]
                a_obj = {"abstract": a}
                b.add_object(
                    collection="DBpediaArticle",
                    properties=a_obj,
                    uuid=generate_uuid5(r["url"]["value"])
                )
        else:
            print(f"No abstract found for initial sparql query.")
except Exception as exception:
    print(f"An error occurred while querying DBpedia: {exception}")

no_articles = dbpedia.aggregate.over_all(total_count=True)
print(no_articles.total_count)


def create_subject_variants(subjects):
    ie = inflect.engine()
    subject_variants = set()

    roman_numeral_pattern = re.compile(r'^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$',
                                       re.IGNORECASE)

    for subject in subjects:
        subject = subject.strip()
        words = subject.split()

        capitalized_words = []
        for word in words:
            if len(word) > 1 and roman_numeral_pattern.match(word):
                capitalized_words.append(word.upper())
            else:
                capitalized_words.append(word.capitalize())

        capitalized_subject = ' '.join(capitalized_words)
        subject_singular = ie.singular_noun(capitalized_subject)

        subject_variants.add(capitalized_subject)
        if subject_singular:
            subject_variants.add(subject_singular)

    return subject_variants


def search_on_dbpedia_n_add_to_weaviate(subject_variants):
    with client.batch.fixed_size(batch_size=100) as batch_search_1:
        for variant in subject_variants:
            print(variant)
            sparql_query = f"""
                SELECT ?entity ?abstract ?url WHERE {{
                    ?entity rdfs:label '{variant}'@en ;
                            dbo:abstract ?abstract ;
                            foaf:isPrimaryTopicOf ?url .
                    FILTER (lang(?abstract) = 'en')
                }}
                LIMIT 1
            """
            sparql.setQuery(sparql_query)

            try:
                results = sparql.query().convert()
                for result in results["results"]["bindings"]:
                    article = result["abstract"]["value"]
                    article_object = {"abstract": article}
                    batch_search_1.add_object(
                        collection="DBpediaArticle",
                        properties=article_object,
                        uuid=generate_uuid5(result["url"]["value"])
                    )
            except Exception as e:
                print(f"An error occurred while querying variant '{variant}': {e}")


def initial_dbpedia_search(query):
    subjects = set()

    try:
        initial_subjects = llm.with_config(configurable={"llm_temperature": 0.0}).invoke(
            f"For a given query, extract all subjects from it. "
            f"Put them in a comma-separated list where each subject is a name suitable for searching on Wikipedia. "
            f"Do not include any bullet points, special characters, or additional formatting. "
            f"The query is: '{query}'."
        )
        if not initial_subjects:
            raise ValueError("Empty response from LLM for subjects")
        for s in initial_subjects.split(", "):
            subjects.add(s)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return ""

    subject_variants = create_subject_variants(subjects)
    print(subject_variants)

    search_on_dbpedia_n_add_to_weaviate(subject_variants)


def additional_dbpedia_search(query):
    subjects = set()

    try:
        associated_subjects = llm.with_config(configurable={"llm_temperature": 0.0}).invoke(
            f"For a given query, extract all relevant associated subjects to it. "
            f"Put them in a comma-separated list where each subject is a name suitable for searching on Wikipedia. "
            f"Do not include any bullet points, special characters, or additional formatting. "
            f"The query is: '{query}'."
        )
        if associated_subjects:
            for list_subjects in associated_subjects.split("\n"):
                for s in list_subjects.split(", "):
                    subjects.add(s)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return ""

    subject_variants = create_subject_variants(subjects)
    print(subject_variants)

    search_on_dbpedia_n_add_to_weaviate(subject_variants)


sources = []
apology_prompt = "Apologize in one sentence for not being able to provide an answer based on what you found."


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/ai', methods=['GET'])
def get_sources():
    return jsonify(sources)


@app.route('/ai/v1', methods=['POST'])
def search_version_1():
    print("---------- Search Version 1 ----------")
    json_content = request.json
    query = json_content.get("query")
    response_data = {
        "query_response": "",
        "status": 0
    }
    sources.clear()

    print("========== Results from Jeopardy Query ==========")
    res = jeopardy.query.near_text(
        query=query,
        distance=0.34,
        return_metadata=MetadataQuery(distance=True)
    )
    for o in res.objects:
        pprint(o.properties)
        print(o.metadata.distance)

    jeopardy_task = (
        f"For a given query, filter these Jeopardy questions and answers to include only those related to the query, "
        f"then try to construct an answer using only the words from these results. If you can do it, return it. "
        f"If not, respond with ''. "
        f"Do not include any bullet points, special characters, or additional formatting. "
        f"The query is: '{query}'."
    )
    jeopardy_response = ""
    if res.objects:
        try:
            response = jeopardy.generate.near_text(
                query=query,
                limit=len(res.objects),
                grouped_task=jeopardy_task,
                grouped_properties=["category", "question", "answer"]
            )
            jeopardy_response = response.generated if response.generated else ""
            for o in response.objects:
                sources.append(
                    f"Category: {o.properties['category']}\n"
                    f"Question: {o.properties['question']}\n"
                    f"Answer: {o.properties['answer']}"
                )
        except Exception as e:
            print(f"Error during Jeopardy response generation: {e}")

    print("========== Initial DBpedia Search ==========")
    initial_dbpedia_search(query)

    print("========== Results from Initial DBpedia Query ==========")
    res = dbpedia.query.near_text(
        query=query,
        distance=0.34,
        return_metadata=MetadataQuery(distance=True)
    )
    for o in res.objects:
        pprint(o.properties)
        print(o.metadata.distance)

    request_satisfied = llm.with_config(configurable={"llm_temperature": 0.0}).invoke(
        f"Is the answer to {query} in {res.objects}? 'Yes' or 'No'."
    )
    print(request_satisfied)
    dbpedia_response = ""
    if request_satisfied == "Yes" or request_satisfied == "yes":
        first_search_task = (
            f"For the given query, construct an answer using the words from these abstracts. "
            f"Do not include any bullet points, special characters, or additional formatting. "
            f"The query is: '{query}'."
        )
        if res.objects:
            try:
                response = dbpedia.generate.near_text(
                    query=query,
                    limit=len(res.objects),
                    grouped_task=first_search_task
                )
                dbpedia_response = response.generated if response.generated else ""
                for o in response.objects:
                    sources.append(o.properties["abstract"])
            except Exception as e:
                print(f"Error during Initial DBpedia response generation: {e}")
    else:
        print("========== Additional DBpedia Search ==========")
        additional_dbpedia_search(query)

        print("========== Results from Additional DBpedia Query ==========")
        res = dbpedia.query.near_text(
            query=query,
            distance=0.34,
            return_metadata=MetadataQuery(distance=True)
        )
        for o in res.objects:
            pprint(o.properties)
            print(o.metadata.distance)

        second_search_task = (
            f"For the given query, try to construct an answer using only the words from these abstracts. Compare "
            f"every abstract to the query and group them to try and answer. If you can do it, return it. If not, "
            f"respond with ''. "
            f"Do not include any bullet points, special characters, or additional formatting. "
            f"The query is: '{query}'."
        )
        if res.objects:
            try:
                response = dbpedia.generate.near_text(
                    query=query,
                    limit=len(res.objects),
                    grouped_task=second_search_task
                )
                dbpedia_response = response.generated if response.generated else ""
                for o in response.objects:
                    sources.append(o.properties["abstract"])
            except Exception as e:
                print(f"Error during Additional DBpedia response generation: {e}")

    print("Jeopardy Response: " + jeopardy_response)
    print("DBpedia Response: " + dbpedia_response)

    if jeopardy_response and dbpedia_response:
        combined_response = llm.invoke(
            f"Construct a short answer to the query by combining the jeopardy response and the dbpedia response. "
            f"If the query asks for a certain number of items, ensure the response contains only that number. "
            f"Do not include any bullet points, special characters, or additional formatting. "
            f"The query is: '{query}'. "
            f"The jeopardy response is: '{jeopardy_response}'. "
            f"The dbpedia response is: '{dbpedia_response}'."
        )
    elif jeopardy_response:
        combined_response = jeopardy_response
    elif dbpedia_response:
        combined_response = dbpedia_response
    else:
        print("Here")
        apology = llm.invoke(apology_prompt)
        response_data["query_response"] = apology if apology else "An error occurred while apologizing."
        response_data["status"] = 500
        return jsonify(response_data)

    response_data["query_response"] = combined_response
    response_data["status"] = 200
    return jsonify(response_data)


@app.route('/ai/v2', methods=['POST'])
@retry(wait_fixed=2000, stop_max_attempt_number=3)
def search_version_2():
    print("---------- Search Version 2 ----------")
    json_content = request.json
    query = json_content.get("query")
    response_data = {
        "query_response": "",
        "status": 0
    }
    sources.clear()

    subjects = set()
    try:
        extracted_subjects = llm.with_config(configurable={"llm_temperature": 0.0}).invoke(
            f"For a given query, extract all subjects and all associated subjects. "
            f"Put them in a comma-separated list, where each subject is a name suitable for searching on Wikipedia. "
            f"To this list, add the nouns in the query and the nouns associated the terms in the query. "
            f"Do not include any bullet points, underscores, special characters, or additional formatting. "
            f"The query is: '{query}'."
        )
        if not extracted_subjects:
            raise ValueError("Empty response from LLM for subjects")
        for s in extracted_subjects.split(", "):
            subjects.add(s)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return ""

    subject_variants = create_subject_variants(subjects)
    print(subject_variants)

    with client.batch.fixed_size(batch_size=100) as batch_search_2:
        for variant in subject_variants:
            print(variant)
            sparql_query_1 = f"""
                SELECT ?entity ?abstract ?url WHERE {{
                    ?entity rdfs:label '{variant}'@en ;
                            dbo:abstract ?abstract ;
                            foaf:isPrimaryTopicOf ?url .
                    FILTER (lang(?abstract) = 'en')
                }}
            """
            sparql.setQuery(sparql_query_1)
            try:
                results = sparql.query().convert()
                for result in results["results"]["bindings"]:
                    abstract = result["abstract"]["value"]
                    article_object = {"abstract": abstract}
                    batch_search_2.add_object(
                        collection="DBpediaArticle",
                        properties=article_object,
                        uuid=generate_uuid5(result["url"]["value"])
                    )
            except Exception as e:
                print(f"An error occurred while querying variant '{variant}': {e}")

        variants_with_underscores = {variant.replace(" ", "_") for variant in subject_variants}
        for variant in variants_with_underscores:
            for term in subject_variants:
                if variant.lower() != term.lower():
                    print(variant + " " + str(term))
                    sparql_query_2 = f"""
                        SELECT ?entity ?abstract ?url WHERE {{
                            ?entity rdf:type dbo:{variant} ;
                                    dbo:abstract ?abstract ;
                                    foaf:isPrimaryTopicOf ?url .
                            FILTER (lang(?abstract) = 'en')
                            FILTER (CONTAINS(?abstract, '{term}') = true)
                        }}
                    """
                    sparql.setQuery(sparql_query_2)
                    try:
                        results = sparql.query().convert()
                        for result in results["results"]["bindings"]:
                            abstract = result["abstract"]["value"]
                            article_object = {"abstract": abstract}
                            batch_search_2.add_object(
                                collection="DBpediaArticle",
                                properties=article_object,
                                uuid=generate_uuid5(result["url"]["value"])
                            )
                    except Exception as e:
                        print(f"An error occurred while querying variant '{variant}': {e}")

        variants_with_no_spaces = {variant.replace(" ", "") for variant in subject_variants}
        for variant in variants_with_no_spaces:
            for term in subject_variants:
                if variant.lower() != term.lower():
                    print(variant + " " + str(term))
                    sparql_query_3 = f"""
                        SELECT ?entity ?abstract ?url WHERE {{
                            ?entity rdf:type schema:{variant} ;
                                    dbo:abstract ?abstract ;
                                    foaf:isPrimaryTopicOf ?url .
                            FILTER (lang(?abstract) = 'en')
                            FILTER (CONTAINS(?abstract, '{term}') = true)
                        }}
                    """
                    sparql.setQuery(sparql_query_3)
                    try:
                        results = sparql.query().convert()
                        for result in results["results"]["bindings"]:
                            abstract = result["abstract"]["value"]
                            article_object = {"abstract": abstract}
                            batch_search_2.add_object(
                                collection="DBpediaArticle",
                                properties=article_object,
                                uuid=generate_uuid5(result["url"]["value"])
                            )
                    except Exception as e:
                        print(f"An error occurred while querying variant '{variant}': {e}")

    total = dbpedia.aggregate.over_all(total_count=True)
    print(total.total_count)

    res = dbpedia.query.near_text(
        query=query,
        distance=0.34,
        limit=30,
        return_metadata=MetadataQuery(distance=True)
    )
    for o in res.objects:
        pprint(o.properties)
        print(o.metadata.distance)

    dbpedia_response = ""
    search_task = (
        f"For the given query, try to construct an answer using only the words from these abstracts. Compare every "
        f"abstract to the query and group them to try and answer. If you can do it, return it. "
        f"If not, respond with ''. "
        f"Do not include any bullet points, special characters, or additional formatting. "
        f"The query is: '{query}'."
    )
    if res.objects:
        response = dbpedia.generate.near_text(
            query=query,
            limit=len(res.objects),
            grouped_task=search_task
        )
        dbpedia_response = response.generated if response.generated else ""
        for o in response.objects:
            sources.append(o.properties["abstract"])

    print("DBpedia response: " + dbpedia_response)
    if dbpedia_response:
        combined_response = dbpedia_response
    else:
        try:
            apology = llm.invoke(apology_prompt)
            response_data["query_response"] = apology if apology else "An error occurred while apologizing."
            response_data["status"] = 500
            return jsonify(response_data)
        except Exception as e:
            print(f"Error during apology generation: {e}")
            response_data["query_response"] = "An error occurred while apologizing."
            response_data["status"] = 500
            return jsonify(response_data)

    response_data["query_response"] = combined_response
    response_data["status"] = 200

    return jsonify(response_data)


@app.route('/ai/v3/sparql', methods=['POST'])
def generate_sparql_query():
    json_content = request.json
    query = json_content.get("query")
    response_data = {
        "query_response": ""
    }
    sources.clear()

    prompt = (
        f"For a given query, create a sparql interrogation that retrieves all the abstracts related to the most "
        f"comprehensive subject/noun of the query. "
        f"It's mandatory to assign abstract with 'dbo:abstract ?abstract' and url with 'foaf:isPrimaryTopicOf ?url'. "
        f"Use FILTER to get only English results. Use another filter to search for a specific term in ?abstract. "
        f"Example 1:\n"
        f"Question: 'What is the tallest mountain in Europe?'\n"
        f"SELECT DISTINCT ?abstract ?url WHERE {{\n"
        f"  ?e rdf:type dbo:Mountain ;\n"
        f"            dbo:abstract ?abstract ;\n"
        f"            foaf:isPrimaryTopicOf ?url .\n"
        f"  FILTER (CONTAINS(?abstract, 'Europe') = true)\n"
        f"  FILTER (lang(?abstract) = 'en')\n"
        f"}}\n"
        f"Example 2:\n"
        f"Question: 'What can you tell me about the Python programming language?'\n"
        f"SELECT DISTINCT ?abstract ?url WHERE {{\n"
        f"  ?e rdf:type dbo:ProgrammingLanguage ;\n"
        f"            dbo:abstract ?abstract ;\n"
        f"            foaf:isPrimaryTopicOf ?url .\n"
        f"  FILTER (CONTAINS(?abstract, 'Python') = true)\n"
        f"  FILTER (lang(?abstract) = 'en')\n"
        f"}}\n"
        f"Keep the interrogation simple, with as few properties/tags as possible. "
        f"The query is: '{query}'."
    )

    sparql_query = llm.with_config(configurable={"llm_temperature": 0.0}).invoke(prompt)
    response_data["query_response"] = sparql_query
    return jsonify(response_data)


def clean_sparql_query(query):
    query = query.strip()
    while not query.lower().startswith("select"):
        query = query[1:].strip()

    # Ensure the query ends with "}" or a number (for "LIMIT number")
    while not (query.endswith("}") or re.search(r"\d+$", query)):
        query = query[:-1].strip()

    return query


@app.route('/ai/v3/dbpedia', methods=['POST'])
def search_with_generated_query():
    json_content = request.json
    query = json_content.get("query")
    sparql_query = json_content.get("generated_sparql")
    response_data = {
        "query_response": "",
        "status": 0
    }

    sparql_query = clean_sparql_query(sparql_query)
    print(sparql_query)
    sparql.setQuery(sparql_query)

    try:
        results = sparql.query().convert()
        if results["results"]["bindings"]:
            with client.batch.fixed_size(batch_size=100) as batch_search_3:
                for result in results["results"]["bindings"]:
                    abstract = result["abstract"]["value"]
                    article_object = {"abstract": abstract}
                    batch_search_3.add_object(
                        collection="DBpediaArticle",
                        properties=article_object,
                        uuid=generate_uuid5(result["url"]["value"])
                    )
            total = dbpedia.aggregate.over_all(total_count=True)
            print(total.total_count)

            res = dbpedia.query.near_text(
                query=query,
                distance=0.30,
                limit=10,
                return_metadata=MetadataQuery(distance=True)
            )
            for o in res.objects:
                pprint(o.properties)
                print(o.metadata.distance)

            dbpedia_response = ""
            search_task = (
                f"For the given query, try to construct an answer using only these results. If you can do it, "
                f"return it. If not, respond with ''. "
                f"Do not include any bullet points, special characters, or additional formatting. "
                f"The query is: '{query}'."
            )
            if res.objects:
                try:
                    response = dbpedia.generate.near_text(
                        query=query,
                        limit=len(res.objects),
                        grouped_task=search_task
                    )
                    dbpedia_response = response.generated if response.generated else ""
                    for o in response.objects:
                        sources.append(o.properties["abstract"])
                except weaviate.exceptions.WeaviateQueryError as e:
                    print(f"Error during DBpedia response generation: {e.message}")

            print("DBpedia response: " + dbpedia_response)
            if dbpedia_response:
                combined_response = dbpedia_response
            else:
                try:
                    apology = llm.invoke(apology_prompt)
                    response_data["query_response"] = apology if apology else "An error occurred while apologizing."
                    response_data["status"] = 500
                    return jsonify(response_data)
                except Exception as e:
                    print(f"Error during apology generation: {e}")
                    combined_response = "An error occurred while apologizing."

            response_data["query_response"] = combined_response
            response_data["status"] = 200
            print(response_data["query_response"])
        else:
            response_data["query_response"] = "No results found! The query was incorrect."
            response_data["status"] = 500
            return jsonify(response_data)

        print("Success")
        return jsonify(response_data)
    except Exception as e:
        print(e)
        response_data["query_response"] = "The generated query was bad or lacking."
        response_data["status"] = 500
        return jsonify(response_data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
