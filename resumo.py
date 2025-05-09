import os
import streamlit as st
import logging
from google.cloud import logging as cloud_logging
from google import genai
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from datetime import (
    date,
    timedelta,
)

from google.genai.types import (
    ApiKeyConfig,
    AuthConfig,
    EnterpriseWebSearch,
    GenerateContentConfig,
    GenerateContentResponse,
    GoogleMaps,
    GoogleSearch,
    LatLng,
    Part,
    Retrieval,
    RetrievalConfig,
    Tool,
    ToolConfig,
    VertexAISearch,
)
import pandas as pd  # Importe a biblioteca pandas para ler o CSV

# configure logging
logging.basicConfig(level=logging.INFO)
# attach a Cloud Logging handler to the root logger
log_client = cloud_logging.Client()
log_client.setup_logging()

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
client = genai.Client(vertexai=True, project="report-summarize", location="us-east1")
MODEL_ID = "gemini-2.0-flash"


@st.cache_resource
def load_models():
    text_model_flash = GenerativeModel("gemini-2.0-flash-001")
    return text_model_flash


def get_gemini_flash_text_response(
    model: GenerativeModel,
    contents: str,
    # generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    VERTEX_AI_SEARCH_PROJECT_ID = "report-summarize"  # @param {type: "string"}
    VERTEX_AI_SEARCH_REGION = "global"  # @param {type: "string"}
    # Replace this with your App (Engine) ID from Vertex AI Search
    VERTEX_AI_SEARCH_APP_ID = "manejo-solo-app_1746782722229"  # @param {type: "string"}

    VERTEX_AI_SEARCH_ENGINE_NAME = f"projects/{VERTEX_AI_SEARCH_PROJECT_ID}/locations/{VERTEX_AI_SEARCH_REGION}/collections/default_collection/engines/{VERTEX_AI_SEARCH_APP_ID}"

    vertex_ai_search_tool = Tool(
        retrieval=Retrieval(
            vertex_ai_search=VertexAISearch(engine=VERTEX_AI_SEARCH_ENGINE_NAME)
        )
    )

    config = GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=max_output_tokens,
        tools=[vertex_ai_search_tool],
    )

    response = client.models.generate_content(
        contents=contents,  # Usando 'contents' aqui, que conterá o prompt completo
        model=MODEL_ID,
        config=config,
    )

    # final_response = []
    # for response in responses:
    #     try:
    #         # st.write(response.text)
    #         final_response.append(response.text)
    #     except IndexError:
    #         # st.write(response)
    #         final_response.append("")
    #         continue
    return response


st.header("MVP Resumo Report", divider="gray")
text_model_flash = load_models()

st.write("Using Gemini Flash - Text only model")

tratamento = st.radio(
    "Qual tratamento foi realizado?",
    (
        "Sem tratamento",
        "Biológico",
        "Fertilizante Químico",
    ),
    index=None,
)

cultura = st.radio(
    "Qual a cultura das amostras",
    (
        "Soja",
        "Cana",
        "Citrus",
        "Tomate",
        "Cafe",
    ),
    index=None,
)

agro_intel = st.file_uploader(
    "Faça o upload do arquivo com os dados de abundância",
    type=["csv"],
    help="O arquivo deve conter os dados de abundância de bactérias, fungos e nematóides.",
)

max_output_tokens = 2048

prompt = ""  # Inicializa o prompt vazio

if agro_intel is not None:
    try:
        df = pd.read_csv(agro_intel)
        csv_content = df.to_string(index=False)  # Converte o DataFrame para uma string
        prompt = f"""Você é um engenheiro agrônomo no Brasil, especializado em análise de dados biológicos
de solo de {cultura}, que teve o tratamento {tratamento}. Com base nos seguintes dados de abundância:\n\n{csv_content}\n\n**e utilizando seu conhecimento e ferramentas de busca para complementar esta informação, gere um relatório com as seguintes informações:**
1. Interpretação dos dados de abundância **presentes neste contexto**, relacionando os organismos encontrados com a cultura das amostras. **Considere também se há informações adicionais relevantes sobre esses organismos e a cultura em bases de conhecimento.**
2. Sugestões de manejo para o tratamento {tratamento} da cultura {cultura}, **considerando os organismos encontrados e as melhores práticas ou informações científicas disponíveis.**
"""
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {e}")


generate_t2t = st.button("Gerar relatório", key="generate_t2t")
if generate_t2t and prompt:
    # st.write(prompt)
    with st.spinner("Gerando seu relatório usando Gemini..."):
        first_tab1, first_tab2 = st.tabs(["Relatório", "Prompt"])
        with first_tab1:
            response = get_gemini_flash_text_response(
                text_model_flash,
                prompt,
                # generation_config=config,
            )
            if response:
                st.write("Seu relatório:")
                st.write(response.text)
                logging.info(response)
        with first_tab2:
            st.text_area("Prompt utilizado:", prompt, height=300)
