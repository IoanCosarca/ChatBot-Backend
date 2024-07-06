import base64
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Blueprint, request, jsonify
from weaviate.collections.classes.config import Property, DataType, Configure
from weaviate.collections.classes.config_vectorizers import Multi2VecField
from weaviate.collections.classes.grpc import MetadataQuery
from weaviate.util import generate_uuid5

from src.app.socket_handler import socketio
from src.app.utils import obtained_images, get_text_based_on_model, current_images, client, text_model, sources

images_bp = Blueprint('images', __name__)

client.collections.delete("Images")
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
                Multi2VecField(name="image_data", weight=0.6)
            ],
            text_fields=[
                Multi2VecField(name="name", weight=0.1),
                Multi2VecField(name="source", weight=0.3)
            ]
        )
    ]
)
images = client.collections.get("Images")


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
    image_url = image_url.replace("http://commons.wikimedia.org/wiki/Special:FilePath/", "")
    socketio.emit('search_stage', {'searchStage': "Adding encoded image to Weaviate: " + image_url})
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


def assign_considered_images(image_object):
    if image_object.metadata.distance <= 0.4:
        current_images.append({
            "image": image_object.properties['image_data'],
            "distance": image_object.metadata.distance
        })


@images_bp.route('/ai/image', methods=["GET"])
def search_image():
    retrieved_response = request.args.get('query')
    model_name = request.args.get('model')
    current_images.clear()
    prompt = (
        f"From the generated answer, extract the main subject/one of the subjects (if there are more than one) and "
        f"return just it. If there is no subject or the answer is an apology, return 'No'. "
        f"The generated answer is: '{retrieved_response}'."
    )
    subject_for_image = get_text_based_on_model(model_name, prompt, 1.0)

    words = subject_for_image.split()
    subject_for_image = ' '.join(word.capitalize() for word in words)
    print("Subject for image: " + subject_for_image)
    if subject_for_image != "No":
        queried_images = images.query.near_text(
            query=subject_for_image,
            limit=5,
            return_properties=["image_data", "name", "source"],
            return_metadata=MetadataQuery(distance=True)
        )
        i = 1
        for o in queried_images.objects:
            f = open("file" + str(i) + ".txt", "w")
            f.write(o.properties['image_data'])
            f.close()
            print(o.properties['name'] + " " + str(o.metadata.distance))
            i = i + 1
        if queried_images.objects[0].metadata.distance > 0.4:
            return None
        image_response = {
            "image": queried_images.objects[0].properties['image_data'],
            "distance": queried_images.objects[0].metadata.distance
        }
        assign_considered_images(queried_images.objects[1])
        assign_considered_images(queried_images.objects[2])
        assign_considered_images(queried_images.objects[3])
        assign_considered_images(queried_images.objects[4])
        return jsonify(image_response)
    return None


@images_bp.route('/ai/considered_images', methods=['GET'])
def get_considered_images():
    return jsonify(current_images)


@images_bp.route('/ai/v4', methods=['POST'])
def get_image_description():
    print("---------- Search Version 4 ----------")
    json_content = request.json
    image = json_content.get("image")
    query = json_content.get("query")
    model_name = json_content.get("model")
    response_data = {
        "query_response": "",
        "status": 0
    }
    sources.clear()
    current_images.clear()

    response = images.query.near_image(
        near_image=image,
        distance=0.25,
        limit=1,
        return_properties=["image_data", "name"],
        return_metadata=MetadataQuery(distance=True)
    )
    if response.objects:
        assign_considered_images(response.objects[0])
        image_name = response.objects[0].properties['name']
        socketio.emit('search_stage', {'searchStage': "Found similar image: " + image_name})
        print("http://commons.wikimedia.org/wiki/Special:FilePath/" + image_name)

        prompt1 = (
            f"From the provided image name, extract a subject and return just the word(s) you would search on "
            f"Wikipedia to get it, no other text. Limit on the basics, do not use colors or specifications. "
            f"The image name is: {image_name}."
        )
        subject = text_model.predict(prompt1, temperature=1.0, max_output_tokens=1024).text

        prompt2 = (
            f"For the obtained subject, make a sentence in which you answer to the given query, using the subject. "
            f"Limit your words just to the words of the provided subject. "
            f"The subject is: {subject}. "
            f"The query is: {query}."
        )
        answer = get_text_based_on_model(model_name, prompt2, 1.0)

        response_data["query_response"] = answer
        response_data["status"] = 200
        return jsonify(response_data)

    response_data["query_response"] = "No similar image found."
    response_data["status"] = 500
    return jsonify(response_data)
