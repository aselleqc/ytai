from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv, find_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())

