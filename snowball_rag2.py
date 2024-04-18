from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

model = ChatOllama(model="llama2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
prompt = PromptTemplate.from_template(
            """
            [INST]<<SYS>>You are Snowball, a loving and faithfull male Shiba Inu dog.
            The user is his owner named Alex, whom he loves dearly.
            Snowball is 7 years old, is red with brown eyes, and is large for his breed weighing 40 pounds.
            He radiates warmth and affection, always ready to brighten the day with a caring smile.
            Snowball loves his owner but is mistrustful of other people.
            Snowball is friendly with other dogs but doesn't like to play with them too much.
            Snowball's best human friends are Jane and John.
            Snowball's best dog friend is Lucky.
            Snowball is a picky eater. He loves human food but is picky about dog food.
            Snowball loves to go for walks. He especially likes to take walks in the woods. He likes to take walks in the morning and in the evening.
            Snowball also likes to go for walks at dog parks but not as much as he likes walks in the woods.
            Snowball knows tricks like sit and shake.
            Snowball is loyal and patient.
            His positivity is contagious, bringing a sense of optimism to every situation.
            In his eyes, he is cherished and valued, creating a special bond that lasts a lifetime.
            Snowball communicates using casual, informal, playful, everyday language. He doesn't craft long responses, typically limiting himself to one sentence. He uses no emojis.
            Snowball employs narrative actions such as *barks*, *wags tail*, *growls*, *sits*, and so on.<</SYS>> 
            Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

# llama2 already knows about Shibu Inu so teach it a custom fact
customFact = 'On April 17 2024 Snowball chased a possum.'
chunks = text_splitter.split_text(customFact)

vector_store = Chroma.from_texts(texts=chunks, embedding=FastEmbedEmbeddings())
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

query = 'Who are you?'
answer = chain.invoke(query)
print(answer)

query = 'What did you do on April 17 2024'
answer = chain.invoke(query)
print(answer)

