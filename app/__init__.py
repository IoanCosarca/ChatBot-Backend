from flask import Flask
from flask_cors import CORS
from weaviate.util import generate_uuid5

from app.images import images_bp
from app.resources import df, dbpedia, resources_bp
from app.search_version1 import search_version1_bp
from app.search_version2 import search_version2_bp
from app.search_version3 import search_version3_bp
from app.utils import client, sparql, obtained_abstracts


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(images_bp)
    app.register_blueprint(resources_bp)
    app.register_blueprint(search_version1_bp)
    app.register_blueprint(search_version2_bp)
    app.register_blueprint(search_version3_bp)

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

    return app
