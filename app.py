from pprint import pprint
from urllib.error import HTTPError

import pandas as pd
import weaviate
from flask import Flask, request, jsonify
from flask_cors import CORS
from weaviate.collections.classes.config import Configure, VectorDistances, Property, DataType
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5

app = Flask(__name__)
CORS(app)


client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    headers={
        "X-PaLM-Api-Key": "ya29.a0AXooCgsSGsw95XGMVIVP0nW8GzEy1_pg8l42kjGlzvC-odDl90-0SJF4MGTTYNd7z1aXf0rmzzQvkIq-W88TNeO16WJKZVxJQh51RorVgy5iNTfpLvFA9kEuNVO26It6Z7WOtqztPREXwtsbr81HusQmfp4-b-uUHSGWstHWdPfaNv7asoNlGktzFt_baNC9Ib8JLkP8CfjJQ7HpmohPlUT1cnG_9ucHEW-4KScaFubRTMIa1LQslXeTdKtYD9JhHXH_7tx4WBZ1yLTUlRE4oLF5ybFUOf931-C0WAfj5BZJ3JqKNAnUm-yRYzD0dIqGVFDYuVUk_80BOU4E80isKUWbGjrmZ5V-wwiJ9y_mATMMqbX_OFpbPZMpbhhz3Q_IDtzh10w6dCx3qBBCe12jUxKxwC21EgUaCgYKAc4SARMSFQHGX2MiYlkixGA5h5YXFaOQ3KPoTQ0422",
    }
)

try:
    url = 'https://raw.githubusercontent.com/IoanCosarca/Jeopardy-Dataset/main/jeopardy_questions.csv'
    df = pd.read_csv(url, sep=',', nrows=500)

    df['value'] = df['value'].fillna(0)
    df['answer'] = df['answer'].astype(str)

    client.collections.delete("JeopardyQuestion")

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
except HTTPError as http_err:
    print("Token for csv expired! " + str(http_err))


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/ai', methods=['POST'])
def ai_post():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    res = jeopardy.query.near_text(
        query=query,
        distance=0.33,
        return_metadata=MetadataQuery(distance=True),
    )
    for o in res.objects:
        pprint(o.properties)
        print(o.metadata.distance)

    response_data = {
        "query_response": ""
    }
    task = f"Provide a coherent single answer if {query} can be answered with these results or an apology stating simply you cannot answer the question based on current knowledge."
    response = jeopardy.generate.near_text(
        query=query,
        limit=len(res.objects),
        grouped_task=task,
        grouped_properties=["category", "question", "answer"]
    )
    if response.generated:
        response_data["query_response"] = response.generated

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
