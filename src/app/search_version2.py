from pprint import pprint

from flask import Blueprint, request, jsonify
from retrying import retry
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5

from src.app import dbpedia
from src.app.images import add_images_to_weaviate
from src.app.socket_handler import socketio
from src.app.utils import sources, text_model, create_subject_variants, client, sparql, obtained_abstracts, \
    construct_search_task_string, get_text_based_on_model, apology_prompt, clean_word, prompt_already_present

search_version2_bp = Blueprint('search_version2', __name__)


@search_version2_bp.route('/ai/v2', methods=['POST'])
@retry(wait_fixed=2000, stop_max_attempt_number=3)
def search_version_2():
    print("---------- Search Version 2 ----------")
    json_content = request.json
    query = json_content.get("query")
    model_name = json_content.get("model")
    response_data = {
        "query_response": "",
        "status": 0
    }
    sources.clear()

    socketio.emit('search_stage', {'searchStage': "Checking if the answer is already stored in Weaviate."})
    aux_sources = []
    result_images = []
    first_response = ""
    res = dbpedia.query.near_text(
        query=query,
        distance=0.37,
        limit=10
    )
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
            result_images.append({
                "name": o.properties["thumbnail"],
                "source": o.properties["url"]
            })
            for image_url in o.properties["image_list"].split(", "):
                result_images.append({
                    "name": image_url,
                    "source": o.properties["url"]
                })
        first_response = response.generated

    request_satisfied = get_text_based_on_model(model_name, prompt_already_present(query, first_response), 1.0)
    request_satisfied = clean_word(request_satisfied)
    socketio.emit('search_stage', {'searchStage': request_satisfied})
    print(request_satisfied)
    if request_satisfied == "Yes" or request_satisfied == "yes":
        for source in aux_sources:
            sources.append(source)
        dbpedia_response = first_response if first_response else ""
        add_images_to_weaviate(result_images)
    else:
        subjects = set()
        try:
            prompt = (
                f"For the given query, construct a comma-separated list of at most 15 entries, with no bullet points, "
                f"containing:\n"
                f"All subjects from the query, each suitable for Wikipedia search.\n"
                f"3 associated subjects, each suitable for Wikipedia search (mandatory).\n"
                f"All nouns from the query.\n"
                f"3 associated nouns (mandatory).\n"
                f"The query is: '{query}'."
            )
            extracted_subjects = text_model.predict(prompt, temperature=1.0, max_output_tokens=1024).text
            if not extracted_subjects:
                raise ValueError("Empty response from LLM for subjects")
            for s in extracted_subjects.split(", "):
                subjects.add(clean_word(s))
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            return ""

        subject_variants = create_subject_variants(subjects)
        print(subject_variants)

        with client.batch.fixed_size(batch_size=100) as batch_search_2:
            for variant in subject_variants:
                socketio.emit('search_stage', {'searchStage': "Searching variant: " + variant})
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

            total = dbpedia.aggregate.over_all(total_count=True).total_count
            socketio.emit('article_count_update', {'articleCount': total})

            variants_with_underscores = {variant.replace(" ", "_") for variant in subject_variants}
            for variant in variants_with_underscores:
                for term in subject_variants:
                    if variant.lower() != term.lower():
                        socketio.emit('search_stage', {'searchStage': "Searching variant: " + variant + ", containing: " + term})
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
                            if results["results"]["bindings"]:
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
                            total = dbpedia.aggregate.over_all(total_count=True).total_count
                            socketio.emit('article_count_update', {'articleCount': total})
                        except Exception as e:
                            print(f"An error occurred while querying term '{term}': {e}")

        res = dbpedia.query.near_text(
            query=query,
            distance=0.37,
            limit=10
        )
        result_images.clear()
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
        apology = get_text_based_on_model(model_name, apology_prompt, 0.0)
        response_data["query_response"] = apology if apology else "An error occurred while apologizing."
        response_data["status"] = 500
        return jsonify(response_data)

    response_data["query_response"] = combined_response
    response_data["status"] = 200

    return jsonify(response_data)
