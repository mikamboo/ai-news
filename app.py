import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_community.llms import Ollama, Bedrock
from langchain_community.chat_models import BedrockChat

# Load environment variables from .env file
load_dotenv()

# os.environ["SERPER_API_KEY"] = ""
# os.environ["OPENAI_API_KEY"] = ""
# os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"
# os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
# os.environ["AWS_ACCESS_KEY_ID"] = ""
# os.environ["AWS_SECRET_ACCESS_KEY"] = ""
# os.environ["AWS_SESSION_TOKEN"] = ""

search_tool   = SerperDevTool()
verbose_mode  = True

# llm = Ollama(model="openhermes", base_url="http://localhost:9000")
# llm = BedrockChat(model_id="mistral.mistral-7b-instruct-v0:2")
# llm = Bedrock(model_id="meta.llama2-13b-v1")
# llm = Bedrock(model_id="amazon.titan-text-lite-v1")
# llm = Bedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
llm = BedrockChat(model_id="anthropic.claude-3-haiku-20240307-v1:0")

# Creating a senior researcher agent with memory and verbose mode
news_collector = Agent(
  role='News Collector',
  goal='Rechercher les dernières nouvelles sur un sujet donné',
  verbose=verbose_mode,
  memory=True,
  backstory=(
    "Passionné par l'information et toujours à l'affût des dernières nouvelles,"
    " cet agent utilise des techniques avancées de recherche pour scanner Internet"
    " et collecter des articles, des blogs et des rapports sur des sujets d'actualité."
  ),
  tools=[search_tool],
  allow_delegation=True,
  llm=llm
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='Rédige des résumés impactants sur {topic}',
  verbose=verbose_mode,
  memory=True,
  backstory=(
    "Tu es un journaliste écrivain talentueux qui sait captiver son audience."
    "Tu es capable de rédiger des résumés impactants sur les sujets les plus importants."
    "Tu es passionné par l\'écriture et tu aimes partager tes connaissances avec le monde."
  ),
  tools=[search_tool],
  allow_delegation=False,
  llm=llm
)


# Research task
research_task = Task(
  description=(
    "Rechercher et collecter les {num_posts} principales actualités sur {topic}"
    " Les date de parution doivent être incluses dans la période du {period}."
    " Focaliser sur les sources fiables et compiler une liste d'articles pertinents."
    " Se limiter aux sites les plus fiables et les plus pertinents."
    " Fournis des liens vers les articles originaux pour plus de détails."
  ),
  expected_output='Une liste d\'articles et de liens sur le sujet {topic}.',
  agent=news_collector,
  llm=llm
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "Rédiger des résumés en {lang} des {num_posts} actualités notables sur {topic} dans la période du {period}."
    " Les résumés doivent être clairs et concis, et doivent captiver l'attention du lecteur."
    " Génère le titre, la catégorie de l'actualité, un résumé court et un résumé long de chaque actualité."
    " Rédige uniquement en {lang}."
    " Fournis des liens vers les articles originaux pour plus de détails."
  ),
  expected_output=(
    '''
    Le résultat doit être sous d'une liste de bloc format markdown suivant:

    ## Titre de la recherche
    
    ### 1. [Titre de l'actualité]

    🏷️ Catégorie de l'actualité
    
    Résumé court de l'actualité (max 50 mots)
    
    Résumé long de l'actualité (max 200 mots)
    
    Soucres:
    - [site web](lien article)
    - [site web](lien article)
    - [site web](lien article)


    ### 2. [Titre de l'actualité]

    🏷️ Catégorie de l'actualité
    
    Résumé court de l'actualité (max 50 mots)
    
    Résumé long de l'actualité (max 200 mots)
    
    Soucres:
    - [site web](lien article)
    - [site web](lien article)
    - [site web](lien article)
    '''
  ),
  agent = writer,
  async_execution = False,
  output_file = f"{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}-news_ai.md"
)


# Forming the tech-focused crew with enhanced configurations
crew = Crew(
  agents=[news_collector, writer],
  tasks=[research_task, write_task],
  process=Process.sequential
)

# Starting the task execution process with enhanced feedback
start_time = time.time()

result = crew.kickoff(inputs={'topic': 'Evènement Digital en France', 'lang': 'français', 'period': '1 au 7 mai 2024', 'num_posts': 3})

end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")
