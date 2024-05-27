from urllib.error import HTTPError

import pandas as pd
import weaviate
import weaviate.classes.config as wc
from flask import Flask, request, jsonify
from weaviate.util import generate_uuid5

app = Flask(__name__)


client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    headers={
        "X-PaLM-Api-Key": "ya29.a0AXooCgskXrcT2H-oiE22-TNJMZx6BgF0McidRb7hkS7_9vB3dpzggrDoL9QHrlUJmHCcoNrAZPHAFXgUsGDfZRMQ2FTRtz6SNhY_cS1KOMuwT5f3u7J3z6_4eND5B7rCuv0jv85YCDOhozRRoVoGvOAZ3BQjzrNw7EG373x3ofVQCcXYQMZcY2tv2U7njh8M2sp0Qwr5RIjoIZHylxVKapacYg7VJxnawqy6Q-69vo0NSw-c8L0Gx4qMIEhYuR0XRiQPsTKEzx4ksw9x8ZXMMrewWtrgCF56_akqCyxJxAbMalt192vQ1N9EXDFmcrX57qIYx1RuAfq_9jF3bcOdds2giwZsjMQh-3Rq22qZmNL-3_6poE0Cv2tMpLaLLCetWoBoIALJ4icnCnzKIcL7jabHBFVZewVbaCgYKAVQSARMSFQHGX2MiXE_Jb_cbG1Cm9YrPi9Pm9w0423",
    }
)

try:
    url = 'https://raw.githubusercontent.com/IoanCosarca/Jeopardy-Dataset/main/jeopardy_questions.csv'
    df = pd.read_csv(url, sep=',', nrows=500)
    print(df.head())

    df['value'] = df['value'].fillna(0)
    df['answer'] = df['answer'].astype(str)
    print(df)

    client.collections.delete("JeopardyQuestion")

    client.collections.create(
        name="JeopardyQuestion",
        properties=[
            wc.Property(name="category", data_type=wc.DataType.TEXT),
            wc.Property(name="question", data_type=wc.DataType.TEXT),
            wc.Property(name="answer", data_type=wc.DataType.TEXT),
        ],
        vectorizer_config=[
            wc.Configure.NamedVectors.text2vec_palm(
                name="category", source_properties=["category"], project_id="t-monument-297120"
            ),
            wc.Configure.NamedVectors.text2vec_palm(
                name="question", source_properties=["question"], project_id="t-monument-297120"
            )
        ],
        generative_config=wc.Configure.Generative.palm(project_id="t-monument-297120"),
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
    print(f"query {query}")

    # Perform the query
    res = jeopardy.query.near_text(
        query=query,
        target_vector="question",
        limit=2
    )

    # Transform the result into a serializable format
    response_data = []
    for obj in res.objects:
        response_data.append({
            "category": obj.properties.get("category"),
            "question": obj.properties.get("question"),
            "answer": obj.properties.get("answer")
        })

    # Return the response as JSON
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
