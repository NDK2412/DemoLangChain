from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Thay Chroma bằng FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import Runnable
import google.generativeai as genai
import os

# Tắt cảnh báo oneDNN từ TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 1. Khởi tạo API Key của Gemini
GOOGLE_API_KEY = "AIzaSyAY19JQe3FEd5uZK95ypTxslXnY7YHdiEk"  # API key của bạn
genai.configure(api_key=GOOGLE_API_KEY)

# 2. Tạo lớp LLM tùy chỉnh cho Gemini, kế thừa từ Runnable
class GeminiLLM(Runnable):
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, dict):
            prompt = input.get("prompt", str(input))
        else:
            prompt = str(input)
        response = self.model.generate_content(prompt)
        return response.text

    def _get_input_type(self):
        return str

    def _get_output_type(self):
        return str

# 3. Tải tất cả file .txt từ một thư mục
folder_path = "D:\\PY_Code\\ChatBox_LC\\Demo_folder\\Text"
loader = DirectoryLoader(
    folder_path,
    glob="*.txt",
    loader_cls=TextLoader
)
documents = loader.load()

# 4. Chia nhỏ tài liệu thành các đoạn (chunks) để embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 5. Tạo embeddings (dùng HuggingFace) và lưu vào FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(texts, embeddings)

# 6. Tạo Prompt Template với yêu cầu trả lời ngắn gọn
prompt_template = """
Bạn là một chatbot trả lời ngắn gọn dựa trên tài liệu .txt sau:

Nội dung tài liệu:
{context}

Câu hỏi: {question}

Trả lời ngắn gọn:
Nếu không liên quan đến tài liệu, trả lời: "Không có thông tin trong tài liệu."
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# 7. Khởi tạo mô hình Gemini và chuỗi RetrievalQA
gemini_llm = GeminiLLM()
qa_chain = RetrievalQA.from_chain_type(
    llm=gemini_llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt}
)

# 8. Hàm chạy chatbot
def run_chatbot():
    print("Chào bạn! Tôi là chatbot dùng Gemini API trả lời dựa trên các file .txt. Hãy đặt câu hỏi hoặc nhập 'exit' để thoát.")
    while True:
        question = input("Câu hỏi của bạn: ")
        if question.lower() == "exit":
            print("Tạm biệt!")
            break
        response = qa_chain.invoke({"query": question})
        print("Trả lời:", response["result"])

# 9. Chạy chatbot
if __name__ == "__main__":
    run_chatbot()