from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from .models import QuestionAnswer
from .serializers import QuestionAnswerSerializer

import os
import cv2
import numpy as np
import joblib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_groq import ChatGroq
from langchain_community.vectorstores.chroma import Chroma

# Load variables from .env file
# load_dotenv('.env')

llamaparse_api_key="llx-KZm8M1tyCNWv1WQfcp9QyDAcbJSh8CcHVfCNvN2MB3UZy7Pq"
groq_api_key="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"

# llamaparse_api_key = os.getenv('LLAMA_API_KEY')
# groq_api_key = os.getenv("GROQ_API_KEY")

 
def load_or_parse_data():
    data_file = "file/post_images/parsed_data.pkl"

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstruction = """ It contains tables.
        Try to be precise while generating the questions and answers. """
        parser = LlamaParse(api_key=llamaparse_api_key,
                            result_type="markdown",
                            parsing_instruction=parsingInstruction,
                            max_timeout=10000,)
        llama_parse_documents = parser.load_data("file/post_images/temp.pdf")


        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents


    print(parsed_data)
    return parsed_data

pd=load_or_parse_data()

 # Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()
    #print(llama_parse_documents[0].text[:300])

    with open('file/post_images/output.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "file/post_images/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

   #loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    #len(docs)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")
    #docs[0]

    # Initialize Embeddings
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create and persist a Chroma vector database from the chunked documents
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_llamaparse1",  # Local mode with in-memory storage only
        collection_name="rag"
    )


    print('Vector DB created successfully !')
    return vs,embed_model


vs, embed_model = create_vector_database()

chat_model = ChatGroq(temperature=0.0,
                      model_name="mixtral-8x7b-32768",
                      api_key=groq_api_key)

vectorstore = Chroma(embedding_function=embed_model,
                      persist_directory="chroma_db_llamaparse1",
                      collection_name="rag")

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

custom_prompt_template = """Use the following pieces of information to answer questions of the user.

Context: {context}
Question: {question}

Only return the helpful content below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
#
prompt = set_custom_prompt()


PromptTemplate(input_variables=['context', 'question'], template=custom_prompt_template)

PromptTemplate(input_variables=['context', 'question'], template='Use the following pieces of information to answer questions of the user.\n\nContext: {context}\nQuestion: {question}\n\nOnly return the helpful content below and nothing else.\nHelpful answer:\n')

qa = RetrievalQA.from_chain_type(llm=chat_model,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 return_source_documents=True,
                                 chain_type_kwargs={"prompt": prompt})

response = qa.invoke({
    "query": "Generate 20 technical interview questions and answers suitable for a candidate with 0 year of experience "
             "in the field, based on the provided content. Include a mix of basic, intermediate, tricky, and logical "
             "questions. Follow a coherent order in the question formation. Provide the source documents "
    })

temp_str = response['result']
lines = temp_str.split("\n")



def save_questions_and_answers(lines):
    for line in lines:
        if 'Answer:' in line:
            question_part = line.split('Answer:')[0]
            answer_part = line.split('Answer:')[1].strip()
            question = question_part.split('?', -1)[0] + '?'
            
            # Save to the model
            qa = QuestionAnswer(question=question_part, answer=answer_part)
            qa.save()

save_questions_and_answers(lines)

# Print the dictionary
# print(interview_questions_answers)

def home(request):
    return HttpResponse(lines)



class QuestionAnswerAPIView(APIView):
    """
    API View to handle saving questions and answers data.
    """
     # 1. List all
    def get(self, request, *args, **kwargs):
        '''
        List all the Crop_data items for given requested user
        '''
        QA = QuestionAnswer.objects.all()
        serializer = QuestionAnswerSerializer(QA, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        """
        Save questions and answers data from provided lines.
        """
        lines = request.data.get('lines', [])
        if not lines:
            return Response({"error": "No data provided"}, status=status.HTTP_400_BAD_REQUEST)

        for line in lines:
            if 'Answer:' in line:
                question_part = line.split('Answer:')[0]
                answer_part = line.split('Answer:')[1].strip()
                question = question_part.split('?', -1)[0] + '?'

                # Save to the model
                qa = QuestionAnswer(question=question, answer=answer_part)
                qa.save()

        return Response({"message": "Data saved successfully"}, status=status.HTTP_201_CREATED)



