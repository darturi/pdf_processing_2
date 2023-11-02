import os
import shutil
# import fitz
import pickle

import openai
from pypdf import PdfMerger
import textract
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import GPT2TokenizerFast

import pandas as pd
import matplotlib.pyplot as plt


class QueryBase:
    def __init__(self, openAI_key):
        self.corpus_len = 0
        self.corpus_folder = "corpus"
        self.file_list = []
        self.master_merged_filename = os.path.join("compute_resources", "master_merged.pdf")
        self.embeddings = OpenAIEmbeddings(openai_api_key=openAI_key)
        self.key = openAI_key
        self.db = None

        # os.environ["OPENAI_API_KEY"] = openAI_key

    def add_document(self, f_name, update_db=True, save_local=False):
        # add new filename to the list of filenames
        full_file_name = os.path.join(self.corpus_folder, f_name)
        self.file_list.append(full_file_name)

        if len(self.file_list) == 1:
            shutil.copy(full_file_name, self.master_merged_filename)
            return
        """
        # update file containing combined pdfs
        result = fitz.open()

        for pdf in [self.master_merged_filename, full_file_name]:
            with fitz.open(pdf) as mfile:
                result.insert_pdf(mfile)

        result.save(self.master_merged_filename)
        """

        merger = PdfMerger()

        for pdf in [self.master_merged_filename, full_file_name]:
            merger.append(pdf)

        merger.write("self.master_merged_filename")
        merger.close()

        if update_db:
            self.create_database(save_local)

    def create_database(self, save_local=False):
        # Convert pdf to text
        document = textract.process(self.master_merged_filename)
        temp_doc_storage = os.path.join("compute_resources", "temp_doc_storage")

        # Save to .txt and reopen
        with open(temp_doc_storage, 'w') as f:
            f.write(document.decode('utf-8'))

        with open(temp_doc_storage, 'r') as f:
            content = f.read()

        # Step 3: Create function to count tokens
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        def count_tokens(content: str) -> int:
            return len(tokenizer.encode(content))

        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=24,
            length_function=count_tokens,
        )

        chunks = text_splitter.create_documents([content])



        """
        # Create a list of token counts
        token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

        # Create a DataFrame from the token counts
        df = pd.DataFrame({'Token Count': token_counts})

        # Create a histogram of the token count distribution
        df.hist(bins=40, )

        # Show the plot
        plt.show()

        """


        self.db = FAISS.from_documents(chunks, self.embeddings)

        if save_local:
            faiss_file_path = os.path.join("compute_resources", "db_file")
            self.db.save_local(faiss_file_path)

            # Pickle embeddings
            embedding_file_path = os.path.join("compute_resources", "embedding.pkl")
            f = open(embedding_file_path, "wb")
            pickle.dump(self.embeddings, f)
            f.close()

    def query_db_ss(self, q):
        return self.db.similarity_search(q, api_key=self.key)



