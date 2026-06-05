from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
import uuid


load_dotenv()
GROQ_API_KEY= os.getenv("GROQ_API_KEY")

#****************Model Building***********************************

chat_model = ChatGroq(model="llama-3.3-70b-versatile",api_key=GROQ_API_KEY,temperature=0.3)

#*****************************Embedding Model**************************
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

#**********************Text splitter*********************************
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

#*********************Vector DB********************************************
CURRENT_COLLECTION = "default_collection"

def create_vector_store(text):
    global CURRENT_COLLECTION
    CURRENT_COLLECTION = f'collection_{uuid.uuid4().hex}'

    chunks =text_splitter.split_text(text)
    vector_store = Chroma.from_texts(texts=chunks,embedding=embedding_model,persist_directory='chroma_db',
                                     collection_name=CURRENT_COLLECTION
                                     )
    vector_store.persist()
    return vector_store

#**********************Load vector store************************
def load_vector_store():
    vector_store = Chroma(persist_directory='chroma_db',embedding_function=embedding_model,collection_name=CURRENT_COLLECTION
                          )
    return vector_store

#******************Function********************

# **************************** 1. Explaination*****************************

def explain_text(vector_store):

    retriever = vector_store.as_retriever(
        search_type='mmr',         # This avoids repetitive chunks...............
        search_kwargs={"k": 4,
                       'fetch_k':10}     # This avoids repetitive chunks.........
    )

    prompt_template = """
    You are an AI teacher.

    ONLY explain using the provided context.

    Context:
    {context}

    Question:
    Explain the chapter in very simple student-friendly language.

    Instructions:
    - Use easy words
    - Explain important concepts
    - Make it beginner friendly
    - Do NOT give unrelated AI/ML explanations
    - Stay strictly within the context

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    response = qa_chain.invoke({'query': "Explain this chapter"})
    print(response['source_documents'])

    return response['result']   # returning result instead of whole dictionary.....

#***************************************2nd Notes Generator*****************************************
def generate_notes(vector_store):
    retriever = vector_store.as_retriever(
        search_type ='mmr',    # improves matching....
        search_kwargs = {'k':5,
                         'fetch_k':10}
    )
    prompt_template = """ 
    Use Only the provided context.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    -Create short and understandable notes
    -Use Bullet points
    -keep notes short
    -Highlight important concept
    -Use simple student-friendly language
    -Stay within Context
    -Don not add unrelated information
    
    Answer:
    """
    PROMPT = PromptTemplate(template= prompt_template,
                            input_variables=['context','question'])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm = chat_model,
        retriever = retriever,
        return_source_documents=True,
        chain_type = 'stuff',
        chain_type_kwargs={
            'prompt':PROMPT
        }
    )
    response = qa_chain.invoke({
        'query': "Generate short notes for this chapter"
    })
    print(response['source_documents'])
    return response['result']

#********************************************* 3rd quiz generator**********************
def quiz_generator(vector_store):
    retriever = vector_store.as_retriever(
        search_type = 'mmr',
        search_kwargs={'k':5,
                       'fetch_k':10})
    
    prompt_template = """
    You are an AI quiz generator.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Instructions:
    -Create 5 quiz questions
    -Include:
       * 3 MCQs
       * 2 True/False questions
    -Also provide correct answers
    -Stay within context
    - Keep questions simple and educational
    - Do not add unrelated questions

    Format:

    Q1:

    Options:
    
    Answer:

    
    Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=['context','question'])
    
    qa_chain = RetrievalQA.from_chain_type(llm=chat_model,
                                           retriever=retriever,
                                           chain_type='stuff',
                                           return_source_documents=True,
                                           chain_type_kwargs={
                                               'prompt':PROMPT})
    response = qa_chain.invoke({'query':'Generate quiz questions from this chaptor'})
    print(response['source_documents'])
    return response['result']

#***************************************** 4th ask question****************************

def ask_question(vector_store,user_question):
    retriever = vector_store.as_retriever(
        search_type = 'mmr',
        search_kwargs={'k':5,
                       'fetch_k':10}
    )

    prompt_template = """
    You are an AI study assistant.
    Use only the provided context.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Instructions:
    -Give short and accurate answers
    - Explain in simple student-friendly language
    -Stay within context
    - Do not generate unrelated information
    -If answer is not found in context, say:
        "Answer not found in the provided text."

    Answer:"""
    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=['context','question'])
    qa_chain = RetrievalQA.from_chain_type(llm=chat_model,
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type='stuff',
                                           chain_type_kwargs={
                                               'prompt':PROMPT
                                           })
    response = qa_chain.invoke({'query' : user_question})
    print(response['source_documents'])
    return response['result']


