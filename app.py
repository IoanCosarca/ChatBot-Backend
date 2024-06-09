import sys
from pprint import pprint

import nltk
import pandas as pd
import weaviate
from SPARQLWrapper import SPARQLWrapper, JSON
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAI
from nltk.stem import WordNetLemmatizer
from weaviate.collections.classes.config import Configure, VectorDistances, Property, DataType
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5

app = Flask(__name__)
CORS(app)

sys.stdout.reconfigure(encoding='utf-8')
nltk.download('wordnet')

PALM_API_KEY = "ya29.a0AXooCgugn07zpeKm4mDWkdVZ-m335ousCLAyYQUSRHaEPs6gA7QBhH4VKZGkBSr0DKlHsqwvjy8NQfxik9CBCcwdFQdFGDOglE4LwnEiGCIDQmscSaD2OkpqO7i8FONWiMYGBIwSr6wnhnqP2OM4b3NAcnbbqGEbZOlHHiFqVp_q-hJk7EtwE98uQLsp-0ZeiWMWUaPHO3_mDlSJvn_vOpRsPwlYaWCb2-kmjIYD5bkLcKNBbTfSrQTbWPnjjgG6I7v0JWH7U9K3zHSwX3jVWZayVvNSoiX9P5TxjhoQ-oLHI-4mUu94qAdsy9bokzaRaOb_49zHQ9JDKNSOZBGhEfcINk2R46I2EXWVXnkm5N6Alu5Od2Q4vnLGK51kIel6Raj04eQSj1d-ieIlWWpZDnoUQvNQRWwoaCgYKAboSARMSFQHGX2MiFdyed0wjAHNIsDTVsk0lcw0423"
GOOGLE_API_KEY = "AIzaSyAJ6mi9i3I5qnEGgwJql4eJc6CZULfcYKU"

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY)
# model = new GoogleVertexAI({
#   model: "code-bison",
#   maxOutputTokens: 2048,
# });

client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    headers={
        "X-PaLM-Api-Key": PALM_API_KEY
    }
)

url = 'https://raw.githubusercontent.com/IoanCosarca/Jeopardy-Dataset/main/jeopardy_questions.csv'
df = pd.read_csv(url, sep=',', nrows=500)

df['value'] = df['value'].fillna(0)
df['answer'] = df['answer'].astype(str)

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

with client.batch.fixed_size(batch_size=200) as batch:
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


def capitalize_permutations(subject):
    words = subject.split()
    permutations = []
    for i in range(1 << len(words)):
        permutation = []
        for j, word in enumerate(words):
            if i & (1 << j):
                permutation.append(word.capitalize())
            else:
                permutation.append(word)
        permutations.append(" ".join(permutation))
    return permutations


def initial_dbpedia_search(query):
    lemmatizer = WordNetLemmatizer()
    subjects = set()

    try:
        initial_subjects = llm.with_config(configurable={"llm_temperature": 0.5}).invoke(
            f"Extract all subjects from the query '{query}'."
            f"Put them in a comma-separated list where each subject is a name suitable for searching on Wikipedia. "
            f"Do not include any bullet points, special characters, or additional formatting."
        )
        if not initial_subjects:
            raise ValueError("Empty response from LLM for subjects")
        for s in initial_subjects.split(", "):
            subjects.add(s)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return ""

    subject_variants = set()
    for subject in subjects:
        subject = subject.strip()
        subject_singular = lemmatizer.lemmatize(subject)
        subject_plural = lemmatizer.lemmatize(subject, 'n')

        capitalized_subject_permutations = capitalize_permutations(subject)

        subject_variants.update(capitalized_subject_permutations)
        subject_variants.add(subject_singular)
        subject_variants.add(subject_plural)

    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    for variant in subject_variants:
        sparql_query = f"""
            SELECT ?abstract WHERE {{
                ?entity rdfs:label "{variant}"@en ;
                        dbo:abstract ?abstract .
                FILTER (lang(?abstract) = 'en')
            }}
            LIMIT 1
            """
        sparql.setQuery(sparql_query)

        try:
            results = sparql.query().convert()
            if results["results"]["bindings"]:
                abstract = results["results"]["bindings"][0]["abstract"]["value"]
                print(f"Abstract found for variant '{variant}': {abstract}")

                article_object = {"abstract": abstract}
                with client.batch.fixed_size(batch_size=200) as batch:
                    batch.add_object(
                        collection="DBpediaArticle",
                        properties=article_object,
                        uuid=generate_uuid5(article_object)
                    )
            else:
                print(f"No abstract found for variant '{variant}'")
        except Exception as e:
            print(f"An error occurred while querying variant '{variant}': {e}")


def additional_dbpedia_search(query):
    lemmatizer = WordNetLemmatizer()
    subjects = set()

    try:
        associated_subjects = llm.with_config(configurable={"llm_temperature": 0.0}).invoke(
            f"Extract relevant associated subjects to the query '{query}'."
            f"Put them in a comma-separated list where each subject is a name suitable for searching on Wikipedia. "
            f"Do not include any bullet points, special characters, or additional formatting."
        )
        if associated_subjects:
            for list_subjects in associated_subjects.split("\n"):
                for s in list_subjects.split(", "):
                    subjects.add(s)
            print(subjects)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return ""

    subject_variants = set()
    for subject in subjects:
        subject = subject.strip()
        subject_singular = lemmatizer.lemmatize(subject)
        subject_plural = lemmatizer.lemmatize(subject, 'n')

        capitalized_subject_permutations = capitalize_permutations(subject)

        subject_variants.update(capitalized_subject_permutations)
        subject_variants.add(subject_singular)
        subject_variants.add(subject_plural)

    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    for variant in subject_variants:
        sparql_query = f"""
            SELECT ?abstract WHERE {{
                ?entity rdfs:label "{variant}"@en ;
                        dbo:abstract ?abstract .
                FILTER (lang(?abstract) = 'en')
            }}
            LIMIT 1
            """
        sparql.setQuery(sparql_query)

        try:
            results = sparql.query().convert()
            if results["results"]["bindings"]:
                abstract = results["results"]["bindings"][0]["abstract"]["value"]
                print(f"Abstract found for variant '{variant}': {abstract}")

                article_object = {"abstract": abstract}
                with client.batch.fixed_size(batch_size=200) as batch:
                    batch.add_object(
                        collection="DBpediaArticle",
                        properties=article_object,
                        uuid=generate_uuid5(article_object)
                    )
            else:
                print(f"No abstract found for variant '{variant}'")
        except Exception as e:
            print(f"An error occurred while querying variant '{variant}': {e}")


sources = []


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/ai', methods=['GET'])
def get_sources():
    return jsonify(sources)


@app.route('/ai', methods=['POST'])
def ai_post():
    print("-------------------------------------------------")
    json_content = request.json
    query = json_content.get("query")
    response_data = {
        "query_response": ""
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
        sources.append(f"Category: {o.properties['category']}\n"
                       f"Question: {o.properties['question']}\n"
                       f"Answer: {o.properties['answer']}"
                       )

    jeopardy_task = (
        f"Filter the following Jeopardy questions and answers to include only those that directly answer the query '{query}',"
        f"then provide a coherent single answer if the query can be answered with these results"
        f"or an empty string if the query cannot be answered based on these."
        f"Do not include any bullet points, special characters, or additional formatting."
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
        except Exception as e:
            print(f"Error during Jeopardy response generation: {e}")

    print("========== Initial DBpedia Search ==========")
    initial_dbpedia_search(query)

    print("========== Results from Initial DBpedia Query ==========")
    res = dbpedia.query.near_text(
        query=query,
        distance=0.4,
        return_metadata=MetadataQuery(distance=True)
    )
    for o in res.objects:
        pprint(o.properties)
        print(o.metadata.distance)
        sources.append(o.properties["abstract"])

    dbpedia_task = (
        f"Provide a coherent single answer if {query} can be answered with these results or an empty string if you"
        f"cannot answer the question based on these."
        f"Do not include any bullet points, special characters or additional formatting."
    )
    dbpedia_response = ""
    if res.objects:
        try:
            response = dbpedia.generate.near_text(
                query=query,
                limit=len(res.objects),
                grouped_task=dbpedia_task
            )
            dbpedia_response = response.generated if response.generated else ""
        except Exception as e:
            print(f"Error during Initial DBpedia response generation: {e}")

    if not dbpedia_response:
        print("========== Additional DBpedia Search ==========")
        additional_dbpedia_search(query)

        print("========== Results from Additional DBpedia Query ==========")
        res = dbpedia.query.near_text(
            query=query,
            distance=0.45,
            return_metadata=MetadataQuery(distance=True)
        )
        for o in res.objects:
            pprint(o.properties)
            print(o.metadata.distance)
            sources.append(o.properties["abstract"])

        if res.objects:
            try:
                response = dbpedia.generate.near_text(
                    query=query,
                    limit=len(res.objects),
                    grouped_task=dbpedia_task
                )
                dbpedia_response = response.generated if response.generated else ""
            except Exception as e:
                print(f"Error during Additional DBpedia response generation: {e}")

    print("Jeopardy Response: " + jeopardy_response)
    print("DBpedia Response: " + dbpedia_response)

    if jeopardy_response and dbpedia_response:
        combined_response = llm.invoke(
            f"Combine the responses '{jeopardy_response}' and '{dbpedia_response}' to provide an answer to the query '{query}'."
            f"Ensure the combined response satisfies the constraints of the query, such as the number of items requested."
            f"Do not include any bullet points, special characters, or additional formatting."
            f"If the query asks for a specific number of items, ensure the response contains exactly that number of items."
        )
    elif jeopardy_response:
        combined_response = jeopardy_response
    elif dbpedia_response:
        combined_response = dbpedia_response
    else:
        try:
            apology_message = llm.invoke(
                "Apologize for not being able to provide an answer based on the current knowledge."
            )
            combined_response = apology_message if apology_message else "An error occurred while processing your request."
        except Exception as e:
            print(f"Error during apology generation: {e}")
            combined_response = "An error occurred while processing your request."

    response_data["query_response"] = combined_response

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
