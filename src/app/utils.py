import re
import sys

import inflect
import vertexai
import weaviate
from SPARQLWrapper import SPARQLWrapper, JSON
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import ChatModel, TextGenerationModel

sys.stdout.reconfigure(encoding='utf-8')

PALM_API_KEY = "ya29.a0AXooCguKIV6yyCyEOwZ0xsCSLyNyQ4I9gOVcwy9MyJ1ztAlWzMZjOYbHyKX1_RDhUKTG5e6WX92Iwww04SVJjtjCcBjp09Z9zi42o_yibUJ_mt3k_QEUCyIcs3um5ZpjaRAPrfCUW01mnd5lhbq-hIE0CSySIE_uhF0dxr4RaIGKCJtw0MW-eYV5snosrW_JZdu0D3bGQMKJO65xi8ZCKdIbnHIW0YJ-z1ArR-RDVPSZ9FmRLHQtpqFurNl-EQlznNjocZBejarkiXlg-3v6tI1IeCNABrQGfx4rYFDrW1ym127xtMOYdpu8qJ_NbLW1ILa5LQQMc8xlzA9QLe4W7Qtr_33yxH7PYdDtNOXdAjtxv0WeSVNTUZEfwCAQoV8MT3F5fot8onjSSfJMrTBPMEdVM67C9Gk2NAaCgYKAZgSARMSFQHGX2MiZhfQM0kWLx3GSmq7uOcwzA0425"

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
        subject_variants.add(capitalized_subject)

        if not subject[0].isupper():
            singular_form = ie.singular_noun(subject)
            if singular_form and singular_form.lower() != subject.lower():
                subject_variants.add(singular_form.capitalize())

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
        f"For the given query, use these obtained abstracts and try to answer in 3 sentences to the query, only if the "
        f"words from the abstracts provide the answer. If you can do it, return the answer. If not, just return ''. "
        f"The query is: '{query}'."
    )
    return search_task


def prompt_already_present(query, response):
    prompt = (
        f"Is the answer to the given query in the given response? Respond with 'Yes' or 'No', just the word. "
        f"The query is: '{query}'. "
        f"The response is: '{response}'."
    )
    return prompt
