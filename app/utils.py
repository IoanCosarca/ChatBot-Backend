import re
import sys

import inflect
import vertexai
import weaviate
from SPARQLWrapper import SPARQLWrapper, JSON
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import ChatModel, TextGenerationModel

sys.stdout.reconfigure(encoding='utf-8')

PALM_API_KEY = "ya29.a0AXooCgs_-6tQqesQ1xdcv7OwE43hhC7UVug5EHOD-cO_shuwunoobN-zsfK-RRgJwRp3D-vFwf-6mvp9tF9hQSVnkUWMUXcA3GlXwZbOsKpwxfYEb7EylLBtXSQtErVIwlsMFZ3BnJcSKq6ss8T8XEvvfGd8zbN1feTIOcek4wg3ZfJrH64d6ovP_18-JVrVfq-IcE1PIhWilkj9tfYehdaYY80bhnZWW_PUdlS_5feeLuXqVOar9hD61aM8ett3rCFCXSneXvfvOEhhShKCmOE7HpGzpLNwEmcBERWRfLsFgnu12otCkh5y9j689ofGnBxiX6C1zQnEC3DSJ2OyjC1QgCWrc4vlbsF66maZERvMzCHUQF3f3O1FssPFxGyJXcjeQj3RoYpxhByTxnbnVCF468bEzpvwo_IaCgYKAS8SARMSFQHGX2MivYo1htFM9Fi8VbjESsVf8g0426"

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

sparql = SPARQLWrapper("https://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

obtained_abstracts = []
obtained_images = []
current_images = []
sources = []
apology_prompt = "Apologize in one sentence for not being able to provide an answer based on what you found."


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


def clean_word(word):
    word = word.strip()
    # Remove any character from the beginning that is not a letter
    word = re.sub(r'^[^a-zA-Z]+', '', word)
    # Remove any character from the end that is not a letter, or, for subjects, ")" or a number
    word = re.sub(r'[^a-zA-Z0-9)]+$', '', word)
    return word


def get_text_based_on_model(model_name, prompt, temperature=0.0):
    text = ""
    if model_name == "chat-bison":
        text = chat_model.start_chat().send_message(prompt, temperature=temperature, max_output_tokens=1024).text
    if model_name == "text-bison":
        text = text_model.predict(prompt, temperature=temperature, max_output_tokens=1024).text
    if model_name == "gemini":
        text = gemini_model.generate_content(prompt).text
    return text


def construct_search_task_string(query):
    search_task = (
        f"For the given query, use these obtained abstracts and try to answer to the query only if the words from "
        f"them provide the answer. If you can do it, return the answer. If not, just return ''. "
        f"Make the generated answer have at most 3 sentences. "
        f"The query is: '{query}'."
    )
    return search_task
