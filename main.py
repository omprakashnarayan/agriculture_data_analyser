from langchain.llms.huggingface_pipeline import HuggingFacePipeline 

from transformers import AutoTokenizer 

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders.csv_loader import CSVLoader 

from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA 

import transformers 

import torch 

import pandas as pd

import gradio 

import textwrap

import chardet

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

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cuda'})

import gradio as gr

import pandas as pd
from langchain_core.documents import Document


def main(dataset, qs):
    # Use chardet to detect the encoding
    with open(dataset.name, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']

    # Load data using the detected encoding
    df = pd.read_csv(dataset.name, encoding=encoding)

    # Get the names of text columns dynamically
    text_columns = [col for col in df.columns if df[col].dtype == 'object']

    # Check if there are any text columns
    if not text_columns:
        raise ValueError("No text columns found in the dataset.")

    # Concatenate text from multiple columns into a single 'text' column
    df['text'] = df[text_columns].apply(lambda x: ' '.join(map(str, x)), axis=1)

    # Create a list of documents
    documents = [Document(page_content=row['text']) for _, row in df.iterrows()]

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    # df = pd.read_csv(dataset.name)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", return_source_documents=False,
                                        retriever=vectorstore.as_retriever())

    # query = "What is the annual salary of Sophie Silva?"

    result = chain(qs)

    wrapped_text = textwrap.fill(result['result'], width=500)
    return wrapped_text



import pandas as pd

def dataset_change(dataset):
    # Use chardet to detect the encoding
    with open(dataset.name, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']

    # Load data using the detected encoding
    df = pd.read_csv(dataset.name, encoding=encoding)

    # Display the first 5 rows in UI
    df_head = df.head(5)

    return df_head

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
    gr.Examples([["What is the Annual Salary of Theodore Dinh?"], ["What is the Department of Parker James?"]], inputs=[qs])

demo.launch(debug=True)