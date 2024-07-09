import re
from pprint import pprint

from flask import Blueprint, request, jsonify
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5

from src.app import dbpedia
from src.app.images import add_images_to_weaviate
from src.app.socket_handler import socketio
from src.app.utils import sources, get_text_based_on_model, sparql, client, obtained_abstracts, \
    construct_search_task_string, gemini_model, apology_prompt

search_version3_bp = Blueprint('search_version3', __name__)


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

    # Ensure the query ends with "}" or "LIMIT <number>"
    while not (query.endswith("}") or re.search(r'(?i)LIMIT\s+\d+$', query)):
        query = query[:-1].strip()

    return query


@search_version3_bp.route('/ai/v3/sparql', methods=['POST'])
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


@search_version3_bp.route('/ai/v3/dbpedia', methods=['POST'])
def search_with_generated_query():
    json_content = request.json
    query = json_content.get("query")
    sparql_query = json_content.get("generated_sparql")
    response_data = {
        "query_response": "",
        "status": 0
    }

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
                        socketio.emit('search_stage', {'searchStage': "Found page: " + url})
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
            total = dbpedia.aggregate.over_all(total_count=True).total_count
            socketio.emit('article_count_update', {'articleCount': total})

            res = dbpedia.query.near_text(
                query=query,
                distance=0.37,
                limit=10
            )
            result_images = []
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
                    result_images.append({
                        "name": o.properties["thumbnail"],
                        "source": o.properties["url"]
                    })
                    for image_url in o.properties["image_list"].split(", "):
                        result_images.append({
                            "name": image_url,
                            "source": o.properties["url"]
                        })
                add_images_to_weaviate(result_images)

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
