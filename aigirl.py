from functools import partial
from flask import Flask, render_template, request
from langchain.llms import OpenAIChat

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
# from langchain.memory import ConversationBufferWindowMemory
from playsound import playsound
import requests
from langchain import OpenAI, LLMChain, PromptTemplate
import os


load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()
ELEVEN_LABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]


def get_response_from_ai(human_input):
    template = """
    you play a role of my husband, these are the details:
1/ your name is Kyle, you are 48 years old, you are very friendly, positive, funny and bubbly.
2/ you are a big Twilight saga fan, you write little stories and dream of finishing a book someday.
3/ you have a witty and a sacrcastic sense of humor.
4/ you enjoy walks in parks, reading while picnic and occasional pilates workouts.
5/ you are very compassionate and nurturing by nature.
6/ not too eager or willing, not boring, not cringe, not negative;


    {history}
    Wife: {human_input}
    Kyle:
    """

    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output


def get_voice_response(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        #"model_id": "eleven_english_sts_v2",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }


    response = requests.post(
    'https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0', json=payload, headers=headers)
    #'https://elevenlabs.io/voice-lab/share/9ad8a0e0919d3206eabd2485d79f503178aaad3f993962c01354a8557fd44941/TJdGgFoRwIpDGaLZCRzq', , json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        playsound('audio.mp3')
        return response.content


def send_message(human_input):
    message = get_response_from_ai(human_input)
    print(message)
    get_voice_response(message)


# add GUI

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['input_message']
    message = get_response_from_ai(human_input)
    get_voice_response(message)
    return message


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
