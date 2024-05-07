# Crew AI News

Use Creaw AI agents to collect and summarize news articles.
You can configure the script to use your favourite LLM (tested with GPT-4 and Bedrock/Claude3)

## Requirement

Set environnment variables (see .env.example file):

- SERPER_API_KEY (serper.dev API key)
- OPENAI_API_KEY (to use GPT models)
- AWS_XXX (to use AWS Bedrock models)

## Install deps

```bash
virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Run app

```bash
python app.py
```

The script will output news summary ans markdown file.
