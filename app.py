import base64
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

import inflect
import pandas as pd
import requests
import vertexai
import weaviate
from SPARQLWrapper import SPARQLWrapper, JSON
from flask import Flask, request, jsonify
from flask_cors import CORS
from retrying import retry
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import ChatModel, TextGenerationModel
from weaviate.collections.classes.config import Configure, VectorDistances, Property, DataType
from weaviate.collections.classes.config_vectorizers import Multi2VecField
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5

app = Flask(__name__)
CORS(app)

sys.stdout.reconfigure(encoding='utf-8')

PALM_API_KEY = "ya29.a0AXooCgtSfB1vjugt4vB6110ztf0BLWPqvdzvXdxGHXFf4g6PVq4fTascyB3xByHOMt5EkJPnvDnDTOmwyL2UavfuA_-h59GVLdhol6j7fD0mZOFbn_LGmnRXUFwUIrNUjPp6f_mMV77TYnjaT_PZOEa_-SLDNtTAmoli5ro19_lWeNGu5nOqvUff9d1EYUXcwv3eKJfIKPImmddnTeErVTKXgyTjkOVaw3oNrEO7pd0hC2qn4ABemBgh7wsoTB62OP0VqA7qxgfAxcctWHY3_djm2lSnY9cnIeKNRr70OJRbMuXOpOWTyrG-NWAXUWq_TrQwDaeMb9zeS0CU7UKUhsnZ1_s4ZJFKBuDfjmcDuzPFr3SkJ9HTLy4uFKazwDjNf-IsPXdheeQQ8yAPsPnHzXbnTXO9T0GgiFoaCgYKAYcSARMSFQHGX2MiTGLld-wWM5M1xFAfhgVIzg0426"

vertexai.init(project="t-monument-297120", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison-32k@002")
text_model = TextGenerationModel.from_pretrained("text-bison@001")
gemini_model = GenerativeModel("gemini-1.0-pro-002")

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

jeopardy_dataset_url = 'https://raw.githubusercontent.com/IoanCosarca/Jeopardy-Dataset/main/jeopardy_questions.csv'
df = pd.read_csv(jeopardy_dataset_url, sep=',', nrows=1000)

client.collections.delete("JeopardyQuestion")
client.collections.delete("Images")
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
print("Jeopardy collection created")

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
print("Added jeopardy questions to Weaviate")

client.collections.create(
    name="Images",
    properties=[
        Property(name="image_data", data_type=DataType.BLOB),
        Property(name="name", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT)
    ],
    vectorizer_config=[
        Configure.NamedVectors.multi2vec_clip(
            name="image",
            image_fields=[
                Multi2VecField(name="image_data", weight=0.4)
            ],
            text_fields=[
                Multi2VecField(name="name", weight=0.1),
                Multi2VecField(name="source", weight=0.5)
            ]
        )
    ]
)
images = client.collections.get("Images")

client.collections.create(
    name="DBpediaArticle",
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE
    ),
    properties=[
        Property(name="abstract", data_type=DataType.TEXT),  # dbo:abstract
        Property(name="url", data_type=DataType.TEXT),
        Property(name="thumbnail", data_type=DataType.TEXT),
        Property(name="image_list", data_type=DataType.TEXT)
    ],
    vectorizer_config=Configure.Vectorizer.text2vec_palm(
        project_id="t-monument-297120",
        model_id="textembedding-gecko@001",
        vectorize_collection_name=False
    ),
    generative_config=Configure.Generative.palm(
        project_id="t-monument-297120",
        model_id="gemini-1.0-pro-002",
        temperature=0.25
    )
)
dbpedia = client.collections.get("DBpediaArticle")
print("Images collection and DBpedia article collection created")

sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

print("Initializing sparql")
sparql_interrogation = f"""
    SELECT DISTINCT ?abstract ?thumbnail (GROUP_CONCAT(?images; separator=", ") AS ?image_list) ?url WHERE {{
        ?entity rdf:type dbo:MilitaryConflict ;
                dbo:date ?date ;
                dbo:abstract ?abstract ;
                dbo:thumbnail ?thumbnail ;
                foaf:depiction ?images ;
                foaf:isPrimaryTopicOf ?url .
        FILTER (?date >= "1900-01-01"^^xsd:date && ?date <= "1999-12-31"^^xsd:date)
        FILTER (CONTAINS(?abstract, 'Europe') = true)
        FILTER (lang(?abstract) = 'en')
    }}
"""
sparql.setQuery(sparql_interrogation)
print("Sparql initialized")

obtained_abstracts = []
obtained_images = []

try:
    with client.batch.fixed_size(batch_size=100) as b:
        initial_results = sparql.query().convert()
        if initial_results["results"]["bindings"]:
            for r in initial_results["results"]["bindings"]:
                a = r["abstract"]["value"]
                obtained_abstracts.append(a)
                u = r["url"]["value"]
                t = r["thumbnail"]["value"]
                images_list = r["image_list"]["value"]
                a_obj = {
                    "abstract": a,
                    "url": u,
                    "thumbnail": t,
                    "image_list": images_list
                }
                b.add_object(
                    collection="DBpediaArticle",
                    properties=a_obj,
                    uuid=generate_uuid5(u)
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


def fetch_and_encode_image(image_url):
    headers = {
        'User-Agent': 'MyCustomAgent/1.0 (ionutcosarca1@gmail.com)'
    }
    try:
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        image_content = response.content
        encoded_image = base64.b64encode(image_content).decode('utf-8')
        return encoded_image
    except requests.RequestException:
        return None


def add_images_to_weaviate(objects):
    with client.batch.fixed_size(batch_size=100) as current_batch:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for obj in objects:
                source = obj.properties["url"]
                thumbnail_url = obj.properties["thumbnail"]
                if thumbnail_url not in obtained_images:
                    obtained_images.append(thumbnail_url)
                    futures.append(executor.submit(fetch_and_encode_and_add, thumbnail_url, source, current_batch))

                image_list = obj.properties["image_list"]
                for image_url in image_list.split(", "):
                    if image_url not in obtained_images:
                        obtained_images.append(image_url)
                        futures.append(executor.submit(fetch_and_encode_and_add, image_url, source, current_batch))

            for future in futures:
                future.result()


def fetch_and_encode_and_add(image_url, source, current_batch):
    encoded_image = fetch_and_encode_image(image_url)
    if encoded_image:
        image_object = {
            "image_data": encoded_image,
            "name": image_url,
            "source": source
        }
        current_batch.add_object(
            collection="Images",
            properties=image_object,
            uuid=generate_uuid5(image_object)
        )


def search_on_dbpedia_n_add_to_weaviate(subject_variants):
    with client.batch.fixed_size(batch_size=100) as batch_search_1:
        for variant in subject_variants:
            print(variant)
            sparql_query = f"""
                SELECT DISTINCT ?abstract ?thumbnail (GROUP_CONCAT(?images; separator=", ") AS ?image_list) ?url WHERE {{
                    ?entity rdfs:label "{variant}"@en ;
                            dbo:abstract ?abstract ;
                            dbo:thumbnail ?thumbnail ;
                            foaf:depiction ?images ;
                            foaf:isPrimaryTopicOf ?url .
                    FILTER (lang(?abstract) = 'en')
                }}
                LIMIT 1
            """
            sparql.setQuery(sparql_query)

            try:
                results = sparql.query().convert()
                for result in results["results"]["bindings"]:
                    abstract = result["abstract"]["value"]
                    if abstract not in obtained_abstracts:
                        obtained_abstracts.append(abstract)
                        url = result["url"]["value"]
                        thumbnail = result["thumbnail"]["value"]
                        image_list = result["image_list"]["value"]
                        article_object = {
                            "abstract": abstract,
                            "url": url,
                            "thumbnail": thumbnail,
                            "image_list": image_list
                        }
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
        prompt = (
            f"For the given query, extract all subjects from it (up to 5 subjects), each suitable for Wikipedia "
            f"search. Output the subjects in a comma-separated format, with no additional text, bullet points, or "
            f"special characters. "
            f"The query is: '{query}'."
        )
        initial_subjects = text_model.predict(prompt, temperature=0.0, max_output_tokens=1024).text
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
        prompt = (
            f"For a given query, extract up to 5 relevant associated subjects, each suitable for Wikipedia search. "
            f"Output the subjects in a comma-separated format, with no additional text, bullet points, or special "
            f"characters. "
            f"The query is: '{query}'."
        )
        associated_subjects = text_model.predict(prompt, temperature=0.0, max_output_tokens=1024).text
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


def get_text_based_on_model(model_name, prompt, temperature=0.0):
    text = ""
    if model_name == "chat-bison":
        text = chat_model.start_chat().send_message(prompt, temperature=temperature, max_output_tokens=1024).text
    if model_name == "text-bison":
        text = text_model.predict(prompt, temperature=temperature, max_output_tokens=1024).text
    if model_name == "gemini":
        text = gemini_model.generate_content(prompt).text
    return text


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/ai', methods=['GET'])
def get_sources():
    return jsonify(sources)


@app.route('/ai/image', methods=["GET"])
def search_image():
    retrieved_response = request.args.get('query')
    model_name = request.args.get('model')
    prompt = (
        f"From the generated answer, extract the main subject/one of the subjects (if there are more than one) and "
        f"return it. If there is no subject or the answer is an apology, return 'No'. "
        f"The generated answer is: '{retrieved_response}'."
    )
    subject_for_image = get_text_based_on_model(model_name, prompt, 1.0)

    words = subject_for_image.split()
    subject_for_image = ' '.join(word.capitalize() for word in words)
    print("Subject for image: " + subject_for_image)
    if subject_for_image != "No":
        response = images.query.near_text(
            query=subject_for_image,
            limit=5,
            return_properties=["image_data", "name"],
            return_metadata=MetadataQuery(distance=True)
        )
        i = 1
        for o in response.objects:
            f = open("file" + str(i) + ".txt", "w")
            f.write(o.properties['image_data'])
            f.close()
            print(o.properties['name'] + " " + str(o.metadata.distance))
            i = i + 1
        image_response = {
            "image": response.objects[0].properties['image_data']
        }
        return jsonify(image_response)
    return None


def construct_search_task_string(query):
    search_task = (
        f"For the given query, use these obtained abstracts and try to answer to the query only if the words from "
        f"them provide the answer. If you can do it, return the answer. If not, just return ''. "
        f"The query is: '{query}'."
    )
    return search_task


@app.route('/ai/v1', methods=['POST'])
@retry(wait_fixed=2000, stop_max_attempt_number=3)
def search_version_1():
    print("---------- Search Version 1 ----------")
    json_content = request.json
    query = json_content.get("query")
    model_name = json_content.get("model")
    response_data = {
        "query_response": "",
        "status": 0
    }
    sources.clear()

    print("========== Results from Jeopardy Query ==========")
    res = jeopardy.query.near_text(
        query=query,
        distance=0.34,
        limit=10
    )
    jeopardy_task = (
        f"From the obtained triplets of category-question-answer, if the words from a triplet can answer the query, "
        f"construct a response using just those words. "
        f"If not, just return ''. "
        f"The query is: '{query}'."
    )
    jeopardy_response = ""
    if res.objects:
        try:
            response = jeopardy.generate.near_text(
                query=query,
                limit=len(res.objects),
                grouped_task=jeopardy_task,
                grouped_properties=["category", "question", "answer"],
                return_metadata=MetadataQuery(distance=True)
            )
            jeopardy_response = response.generated if response.generated else ""
            for o in response.objects:
                pprint(o.properties)
                print(o.metadata.distance)
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
    total = dbpedia.aggregate.over_all(total_count=True)
    print(total.total_count)
    res = dbpedia.query.near_text(
        query=query,
        limit=10
    )
    first_response = ""
    aux_sources = []
    if res.objects:
        response = dbpedia.generate.near_text(
            query=query,
            limit=len(res.objects),
            grouped_task=construct_search_task_string(query),
            grouped_properties=["abstract"],
            return_metadata=MetadataQuery(distance=True)
        )
        for o in response.objects:
            pprint(o.properties)
            print(o.metadata.distance)
            aux_sources.append(o.properties["abstract"])
        first_response = response.generated
        add_images_to_weaviate(response.objects)

    prompt = (
        f"Is the answer to {query} in {first_response}? Respond just with 'Yes' or 'No'."
    )
    request_satisfied = get_text_based_on_model(model_name, prompt, 1.0)
    print(request_satisfied)
    dbpedia_response = ''
    if request_satisfied == "Yes" or request_satisfied == "yes":
        for source in aux_sources:
            sources.append(source)
        dbpedia_response = first_response if first_response else ""
    else:
        print("========== Additional DBpedia Search ==========")
        additional_dbpedia_search(query)

        print("========== Results from Additional DBpedia Query ==========")
        total = dbpedia.aggregate.over_all(total_count=True)
        print(total.total_count)
        res = dbpedia.query.near_text(
            query=query,
            limit=15
        )
        if res.objects:
            response = dbpedia.generate.near_text(
                query=query,
                limit=len(res.objects),
                grouped_task=construct_search_task_string(query),
                grouped_properties=["abstract"],
                return_metadata=MetadataQuery(distance=True)
            )
            dbpedia_response = response.generated if response.generated else ""
            for o in response.objects:
                pprint(o.properties)
                print(o.metadata.distance)
                sources.append(o.properties["abstract"])
            add_images_to_weaviate(response.objects)

    print("Jeopardy Response: " + jeopardy_response)
    print("DBpedia Response: " + dbpedia_response)

    if jeopardy_response and dbpedia_response:
        print("We got here")
        prompt = (
            f"Combine jeopardy response and dbpedia response to provide a single answer for the query if the words in "
            f"them actually provide the answer and are not an apology. If that's not the case, apologize for not being "
            f"able to respond. "
            f"If the query asks for a certain number of items, ensure the response contains only that number. "
            f"The query is: '{query}'. "
            f"The jeopardy response is: '{jeopardy_response}'. "
            f"The dbpedia response is: '{dbpedia_response}'."
        )
        combined_response = get_text_based_on_model(model_name, prompt, 0.0)
    elif jeopardy_response:
        combined_response = jeopardy_response
    elif dbpedia_response:
        combined_response = dbpedia_response
    else:
        apology = gemini_model.generate_content(apology_prompt).text
        response_data["query_response"] = apology if apology else "An error occurred while apologizing."
        response_data["status"] = 500
        return jsonify(response_data)

    print("Combined response: " + combined_response)
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
        prompt = (
            f"For the given query, follow these steps:\n"
            f"1. Extract all the subjects from the query, each suitable for Wikipedia search.\n"
            f"2. Add 3 associated subjects that are relevant to the query, each suitable for Wikipedia search.\n"
            f"3. Extract all nouns from the query.\n"
            f"4. Add 3 associated nouns that are relevant to the query.\n"
            f"Take 15 subjects and nouns in total and output them in a comma-separated format, with no additional "
            f"text, bullet points, or special characters.\n"
            f"The query is: '{query}'."
        )
        extracted_subjects = text_model.predict(prompt, temperature=0.0, max_output_tokens=1024).text
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
                SELECT DISTINCT ?abstract ?thumbnail (GROUP_CONCAT(?images; separator=", ") AS ?image_list) ?url WHERE {{
                    ?entity rdfs:label "{variant}"@en ;
                            dbo:abstract ?abstract ;
                            dbo:thumbnail ?thumbnail ;
                            foaf:depiction ?images ;
                            foaf:isPrimaryTopicOf ?url .
                    FILTER (lang(?abstract) = 'en')
                }}
            """
            sparql.setQuery(sparql_query_1)
            try:
                results = sparql.query().convert()
                for result in results["results"]["bindings"]:
                    abstract = result["abstract"]["value"]
                    if abstract not in obtained_abstracts:
                        obtained_abstracts.append(abstract)
                        url = result["url"]["value"]
                        thumbnail = result["thumbnail"]["value"]
                        image_list = result["image_list"]["value"]
                        article_object = {
                            "abstract": abstract,
                            "url": url,
                            "thumbnail": thumbnail,
                            "image_list": image_list
                        }
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
                        SELECT DISTINCT ?abstract ?thumbnail (GROUP_CONCAT(?images; separator=", ") AS ?image_list) ?url WHERE {{
                            ?entity rdf:type dbo:{variant} ;
                                    dbo:abstract ?abstract ;
                                    dbo:thumbnail ?thumbnail ;
                                    foaf:depiction ?images ;
                                    foaf:isPrimaryTopicOf ?url .
                            FILTER (lang(?abstract) = 'en')
                            FILTER (CONTAINS(?abstract, "{term}") = true)
                        }}
                    """
                    sparql.setQuery(sparql_query_2)
                    try:
                        results = sparql.query().convert()
                        for result in results["results"]["bindings"]:
                            abstract = result["abstract"]["value"]
                            if abstract not in obtained_abstracts:
                                obtained_abstracts.append(abstract)
                                url = result["url"]["value"]
                                thumbnail = result["thumbnail"]["value"]
                                image_list = result["image_list"]["value"]
                                article_object = {
                                    "abstract": abstract,
                                    "url": url,
                                    "thumbnail": thumbnail,
                                    "image_list": image_list
                                }
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
        limit=15
    )
    dbpedia_response = ""
    if res.objects:
        response = dbpedia.generate.near_text(
            query=query,
            limit=len(res.objects),
            grouped_task=construct_search_task_string(query),
            grouped_properties=["abstract"],
            return_metadata=MetadataQuery(distance=True)
        )
        dbpedia_response = response.generated if response.generated else ""
        for o in response.objects:
            pprint(o.properties)
            print(o.metadata.distance)
            sources.append(o.properties["abstract"])
        add_images_to_weaviate(response.objects)

    print("DBpedia response: " + dbpedia_response)
    if dbpedia_response:
        combined_response = dbpedia_response
    else:
        apology = gemini_model.generate_content(apology_prompt).text
        response_data["query_response"] = apology if apology else "An error occurred while apologizing."
        response_data["status"] = 500
        return jsonify(response_data)

    response_data["query_response"] = combined_response
    response_data["status"] = 200

    return jsonify(response_data)


def verify_generated_interrogation(generated_interrogation):
    if generated_interrogation.find("?abstract") == -1:
        return False
    if generated_interrogation.find("?thumbnail") == -1:
        return False
    if generated_interrogation.find("?image_list") == -1:
        return False
    if generated_interrogation.find("?url") == -1:
        return False
    if generated_interrogation.find("(GROUP_CONCAT(?images; separator=', ') AS ?image_list)") == -1:
        return False
    if generated_interrogation.find("dbo:abstract ?abstract") == -1:
        return False
    if generated_interrogation.find("dbo:thumbnail ?thumbnail") == -1:
        return False
    if generated_interrogation.find("foaf:depiction ?images") == -1:
        return False
    if generated_interrogation.find("foaf:isPrimaryTopicOf ?url") == -1:
        return False
    return True


def clean_sparql_query(query):
    query = query.strip()
    while not query.lower().startswith("select"):
        query = query[1:].strip()

    # Ensure the query ends with "}" or a number (for "LIMIT number")
    while not (query.endswith("}") or re.search(r"\d+$", query)):
        query = query[:-1].strip()

    return query


@app.route('/ai/v3/sparql', methods=['POST'])
def generate_sparql_query():
    json_content = request.json
    query = json_content.get("query")
    model_name = json_content.get("model")
    generation_type = json_content.get("type")
    response_data = {
        "query_response": ""
    }
    sources.clear()

    simple_prompt = (
        f"For a given query, write a sparql interrogation that will extract the abstracts related to it that might "
        f"answer it. "
        f"It is mandatory to have the following lines of code in what you generate:\n"
        f"'SELECT DISTINCT ?abstract ?thumbnail (GROUP_CONCAT(?images; separator=', ') AS ?image_list) ?url WHERE {{', "
        f"'dbo:abstract ?abstract ;', "
        f"'dbo:thumbnail ?thumbnail ;', "
        f"'foaf:depiction ?images ;', "
        f"'foaf:isPrimaryTopicOf ?url .' and "
        f"'FILTER (lang(?abstract) = 'en')'. "
        f"Return just the Sparql interrogation. "
        f"The query is: '{query}'."
    )
    prompt_with_examples = (
        f"For a given query, write a sparql interrogation based on the following examples. "
        f"Example 1:\n"
        f"'What is the tallest mountain in Europe?'\n"
        f"SELECT DISTINCT ?abstract ?thumbnail (GROUP_CONCAT(?images; separator=', ') AS ?image_list) ?url WHERE {{\n"
        f"      ?e rdf:type dbo:Mountain ;\n"
        f"            dbo:abstract ?abstract ;\n"
        f"            dbo:thumbnail ?thumbnail ;\n"
        f"            foaf:depiction ?images ;\n"
        f"            foaf:isPrimaryTopicOf ?url .\n"
        f"      FILTER (CONTAINS(?abstract, 'Europe') = true)\n"
        f"      FILTER (lang(?abstract) = 'en')\n"
        f"}}\n"
        f"Example 2:\n"
        f"'What can you tell me about the Python programming language?'\n"
        f"SELECT DISTINCT ?abstract ?thumbnail (GROUP_CONCAT(?images; separator=', ') AS ?image_list) ?url WHERE {{\n"
        f"      ?e rdf:type dbo:ProgrammingLanguage ;\n"
        f"            dbo:abstract ?abstract ;\n"
        f"            dbo:thumbnail ?thumbnail ;\n"
        f"            foaf:depiction ?images ;\n"
        f"            foaf:isPrimaryTopicOf ?url .\n"
        f"      FILTER (CONTAINS(?abstract, 'Python') = true)\n"
        f"      FILTER (lang(?abstract) = 'en')\n"
        f"}}\n"
        f"It's mandatory to assign ?abstract, ?thumbnail, ?image_list and ?url as in the examples. "
        f"Return just the Sparql interrogation. "
        f"The query is: '{query}'."
    )

    valid = False
    sparql_query = ""
    while not valid:
        if generation_type == "with-examples":
            sparql_query = get_text_based_on_model(model_name, prompt_with_examples, 0.0)
        if generation_type == "without-examples":
            sparql_query = get_text_based_on_model(model_name, simple_prompt, 0.0)
        print("------------------------------------")
        print(sparql_query)
        valid = verify_generated_interrogation(sparql_query)

    sparql_query = clean_sparql_query(sparql_query)
    response_data["query_response"] = sparql_query
    return jsonify(response_data)


@app.route('/ai/v3/dbpedia', methods=['POST'])
def search_with_generated_query():
    json_content = request.json
    query = json_content.get("query")
    sparql_query = json_content.get("generated_sparql")
    response_data = {
        "query_response": "",
        "status": 0
    }

    print(sparql_query)
    sparql.setQuery(sparql_query)
    try:
        results = sparql.query().convert()
        if results["results"]["bindings"]:
            with client.batch.fixed_size(batch_size=100) as batch_search_3:
                for result in results["results"]["bindings"]:
                    abstract = result["abstract"]["value"]
                    if abstract not in obtained_abstracts:
                        obtained_abstracts.append(abstract)
                        url = result["url"]["value"]
                        thumbnail = result["thumbnail"]["value"]
                        image_list = result["image_list"]["value"]
                        article_object = {
                            "abstract": abstract,
                            "url": url,
                            "thumbnail": thumbnail,
                            "image_list": image_list
                        }
                        batch_search_3.add_object(
                            collection="DBpediaArticle",
                            properties=article_object,
                            uuid=generate_uuid5(result["url"]["value"])
                        )
            total = dbpedia.aggregate.over_all(total_count=True)
            print(total.total_count)

            res = dbpedia.query.near_text(
                query=query,
                limit=10
            )
            dbpedia_response = ""
            if res.objects:
                response = dbpedia.generate.near_text(
                    query=query,
                    limit=len(res.objects),
                    grouped_task=construct_search_task_string(query),
                    grouped_properties=["abstract"],
                    return_metadata=MetadataQuery(distance=True)
                )
                dbpedia_response = response.generated if response.generated else ""
                for o in response.objects:
                    pprint(o.properties)
                    print(o.metadata.distance)
                    sources.append(o.properties["abstract"])
                add_images_to_weaviate(response.objects)

            print("DBpedia response: " + dbpedia_response)
            if dbpedia_response:
                combined_response = dbpedia_response
            else:
                apology = gemini_model.generate_content(apology_prompt).text
                response_data["query_response"] = apology if apology else "An error occurred while apologizing."
                response_data["status"] = 500
                return jsonify(response_data)

            response_data["query_response"] = combined_response
            response_data["status"] = 200
            print(response_data["query_response"])
        else:
            response_data["query_response"] = "No results found! The query was incorrect."
            response_data["status"] = 500
            return jsonify(response_data)

        return jsonify(response_data)
    except Exception as e:
        print(e)
        response_data["query_response"] = "The generated query was not syntactically correct."
        response_data["status"] = 500
        return jsonify(response_data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
