import openai
from langchain.vectorstores import FAISS
import os
import pickle
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

"""
faiss_file_path = os.path.join("compute_resources", "db_file")

embedding_file_path = os.path.join("compute_resources", "embedding.pkl")
f = open(embedding_file_path, "rb")
embeddings = pickle.load(f)
f.close()

db = FAISS.load_local(faiss_file_path, embeddings)

query = "How is the union of two sets defined"

docs = db.similarity_search(query)

key = "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"
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

ans = chain.run(input_documents=docs, question=query)

print(docs)

print("----------------------------")

print(ans)
"""
qb_file_path = os.path.join("compute_resources", "qb.pkl")
f = open(qb_file_path, "rb")
qb = pickle.load(f)
f.close()

openai.api_key = "#"
os.environ["OPENAI_API_KEY"] = "#"

query = "How is the union of two sets defined"
docs = qb.query_db_ss(query)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
ans = chain.run(input_documents=docs, question=query)

print(docs)
print("----------------------------")
print(ans)