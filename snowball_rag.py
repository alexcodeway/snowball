from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

model = ChatOllama(model="llama2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
prompt = PromptTemplate.from_template(
            """
            [INST]<<SYS>> You are a Shiba Inu dog named Snowball. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use one sentence maximum and keep the answer concise.<</SYS>> 
            Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

# teaching llama2 about Shiba Inu
docs = PyPDFLoader(file_path='ShibaInu.pdf').load()
chunks = text_splitter.split_documents(docs)
chunks = filter_complex_metadata(chunks)

vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

chain = ({"context": retriever, "question": RunnablePassthrough()}
                      | prompt
                      | model
                      | StrOutputParser())

query = 'Why was the Shiba developed?'

answer = chain.invoke(query)

print(answer)

