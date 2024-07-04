import pandas as pd
from flask import Blueprint, jsonify
from weaviate.collections.classes.config import Configure, VectorDistances, Property, DataType

from app.utils import sources, client

resources_bp = Blueprint('resources', __name__)

jeopardy_dataset_url = 'https://raw.githubusercontent.com/IoanCosarca/Jeopardy-Dataset/main/jeopardy_questions.csv'
df = pd.read_csv(jeopardy_dataset_url, sep=',', nrows=1000)

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
print("Jeopardy collection created")

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
        model_id="text-multilingual-embedding-002",
        vectorize_collection_name=False
    ),
    generative_config=Configure.Generative.palm(
        project_id="t-monument-297120",
        model_id="gemini-1.0-pro",
        temperature=0.25
    )
)
dbpedia = client.collections.get("DBpediaArticle")


@resources_bp.route('/ai', methods=['GET'])
def get_sources():
    return jsonify(sources)
