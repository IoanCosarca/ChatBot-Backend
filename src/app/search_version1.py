from pprint import pprint

from flask import Blueprint, request, jsonify
from retrying import retry
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5

from src.app.images import add_images_to_weaviate
from src.app.resources import jeopardy, dbpedia
from src.app.socket_handler import socketio
from src.app.utils import text_model, create_subject_variants, sources, construct_search_task_string, \
    get_text_based_on_model, apology_prompt, client, sparql, obtained_abstracts, clean_word, prompt_already_present

search_version1_bp = Blueprint('search_version1', __name__)


def search_on_dbpedia_n_add_to_weaviate(subject_variants):
    with client.batch.fixed_size(batch_size=100) as batch_search_1:
        for variant in subject_variants:
            socketio.emit('search_stage', {'searchStage': "Searching variant: " + variant})
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
            f"For the given query, construct a comma-separated list of at most 5 entries, containing all subjects from "
            f"the query, each suitable for Wikipedia search. "
            f"The query is: '{query}'."
        )
        initial_subjects = text_model.predict(prompt, temperature=1.0, max_output_tokens=1024).text
        if not initial_subjects:
            raise ValueError("Empty response from LLM for subjects")
        for list_subjects in initial_subjects.split("\n"):
            for s in list_subjects.split(", "):
                subjects.add(clean_word(s))
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
            f"For the given query, construct a comma-separated list of at most 5 entries, containing relevant "
            f"associated subjects to the query, each suitable for Wikipedia search. "
            f"The query is: '{query}'."
        )
        associated_subjects = text_model.predict(prompt, temperature=1.0, max_output_tokens=1024).text
        for list_subjects in associated_subjects.split("\n"):
            for s in list_subjects.split(", "):
                subjects.add(clean_word(s))
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return ""

    subject_variants = create_subject_variants(subjects)
    print(subject_variants)

    search_on_dbpedia_n_add_to_weaviate(subject_variants)


@search_version1_bp.route('/ai/v1', methods=['POST'])
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
        distance=0.37,
        limit=10
    )
    jeopardy_task = (
        f"From the obtained triplets of category-question-answer, if the words from a triplet can answer the query, "
        f"construct a response using just those words. "
        f"If not, just return an empty string ''. "
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
        print("========== Initial DBpedia Search ==========")
        initial_dbpedia_search(query)

        print("========== Results from Initial DBpedia Query ==========")
        total = dbpedia.aggregate.over_all(total_count=True).total_count
        socketio.emit('article_count_update', {'articleCount': total})
        aux_sources.clear()
        result_images.clear()
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
        print(request_satisfied)
        dbpedia_response = ''
        if request_satisfied == "Yes" or request_satisfied == "yes":
            for source in aux_sources:
                sources.append(source)
            dbpedia_response = first_response if first_response else ""
            add_images_to_weaviate(result_images)
        else:
            result_images.clear()
            print("========== Additional DBpedia Search ==========")
            additional_dbpedia_search(query)

            print("========== Results from Additional DBpedia Query ==========")
            total = dbpedia.aggregate.over_all(total_count=True).total_count
            socketio.emit('article_count_update', {'articleCount': total})
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

    print("Jeopardy Response: " + jeopardy_response)
    print("DBpedia Response: " + dbpedia_response)

    if jeopardy_response and dbpedia_response:
        prompt = (
            f"Combine in 3 sentences the jeopardy response and dbpedia response to answer the query, if they are not "
            f"an apology. If that's not the case, apologize for not being able to respond. "
            f"If the query asks for a certain number of items, ensure the response contains only that number. "
            f"The jeopardy response is: '{jeopardy_response}'. "
            f"The dbpedia response is: '{dbpedia_response}'. "
            f"The query is: '{query}'."
        )
        combined_response = get_text_based_on_model(model_name, prompt, 1.0)
    elif jeopardy_response:
        prompt = (
            f"If the jeopardy response is not an empty string or apology, make sure it has only 3 sentences. If "
            f"however it is, apologize for not being able to respond. "
            f"The jeopardy response is: '{jeopardy_response}'."
        )
        combined_response = get_text_based_on_model(model_name, prompt, 1.0)
    elif dbpedia_response:
        prompt = (
            f"If the dbpedia response is not an empty string or apology, make sure it has only 3 sentences. If however "
            f"it is, apologize for not being able to respond. "
            f"The dbpedia response is: '{dbpedia_response}'."
        )
        combined_response = get_text_based_on_model(model_name, prompt, 1.0)
    else:
        apology = get_text_based_on_model(model_name, apology_prompt, 0.0)
        response_data["query_response"] = apology if apology else "An error occurred while apologizing."
        response_data["status"] = 500
        return jsonify(response_data)

    print("Combined response: " + combined_response)
    response_data["query_response"] = combined_response
    response_data["status"] = 200
    return jsonify(response_data)
