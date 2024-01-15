!pip
install
transformers
einops
accelerate
langchain
bitsandbytes
sentence_transformers
faiss - cpu
gradio
pypdf
sentencepiece
# %%
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import AutoTokenizer

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders.csv_loader import CSVLoader

from langchain.vectorstores import FAISS

from langchain.chains import RetrievalQA

import transformers

import torch

import gradio

import textwrap

# %%
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(

    "text-generation",  # task

    model=model,

    tokenizer=tokenizer,

    torch_dtype=torch.bfloat16,

    trust_remote_code=True,

    device_map="auto",

    max_length=1000,

    do_sample=True,

    top_k=10,

    num_return_sequences=1,

    eos_token_id=tokenizer.eos_token_id

)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
# %%
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
# %%
loader = CSVLoader(
    r'C:\Users\Om\PycharmProjects\Agriculture Data Analyser\Data\Production_Crops_Livestock_E_All_Data.csv',
    encoding="ISO-8859-1", csv_args={'delimiter': ','})

data = loader.load()
# %%
vectorstore = FAISS.from_documents(data, embeddings)
# %%
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", return_source_documents=True,
                                    retriever=vectorstore.as_retriever())

query = "What is the annual salary of Sophie Silva?"

result = chain(query)

wrapped_text = textwrap.fill(result['result'], width=500)

wrapped_text
llm("How old is Data Science Dojo?")
# %%
import gradio as gr

import pandas as pd


def main(dataset, qs):
    # df = pd.read_csv(dataset.name)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", return_source_documents=False,
                                        retriever=vectorstore.as_retriever())

    # query = "What is the annual salary of Sophie Silva?"

    result = chain(qs)

    wrapped_text = textwrap.fill(result['result'], width=500)
    return wrapped_text


# %%

# %%
def dataset_change(dataset):
    global vectorstore

    loader = CSVLoader(dataset.name, encoding="ISO-8859-1", csv_args={'delimiter': ','})

    data = loader.load()

    vectorstore = FAISS.from_documents(data, embeddings)

    df = pd.read_csv(dataset.name)

    return df.head(5)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            data = gr.File()

            qs = gr.Text(label="Input Question")

            submit_btn = gr.Button("Submit")

        with gr.Column():
            answer = gr.Text(label="Output Answer")

    with gr.Row():
        dataframe = gr.Dataframe()

    submit_btn.click(main, inputs=[data, qs], outputs=[answer])

    data.change(fn=dataset_change, inputs=data, outputs=[dataframe])

    gr.Examples([["What is the Annual Salary of Theodore Dinh?"], ["What is the Department of Parker James?"]],
                inputs=[qs])

demo.launch(debug=True)

# %%
