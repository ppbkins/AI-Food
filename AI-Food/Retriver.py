
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from pymongo import MongoClient
from langchain_core.documents import Document

load_dotenv()

# Tải dữ liệu từ MongoDB và tạo vector store
def load_data_from_mongo():
    # Kết nối vào DB mà người A đã tạo
    client = MongoClient("mongodb://localhost:27017/")
    collection = client["FoodyDB"]["HaNoiRestaurants"]
    raw_data = collection.find()
    documents = []
    for item in raw_data:
        # Nội dung chính dùng để so sánh vector (Tên + Địa chỉ)
        page_content = f"Quán ăn: {item['Name']}. Địa chỉ: {item['Address']}."
        metadata = {
            "source": "foody",
            "name": item['Name'],
            "address": item['Address'], 
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    #  khởi tạo bộ nhúng
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Lưu vào Vector Store (ChromaDB)
    # Dữ liệu sẽ được lưu vào thư mục 'db_foody' để không bị mất khi tắt máy
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./db_foody"
    )
    print(f"--- Đã nạp thành công {len(documents)} quán ăn vào Vector Store ---")
    return vector_store
    
#cấu trúc dữ liệu mong muốn sau khi xử lý
class ProcessedQuery(BaseModel):
    rewritten_query: str = Field(description="Câu hỏi đã được tối ưu hóa để tìm kiếm vector")
    keywords: List[str] = Field(description="Danh sách các từ khóa món ăn, nguyên liệu")
    filters: Optional[dict] = Field(description="Các bộ lọc như: price_max, location, calories")

def query_processor(user_raw_query: str, llm_model):

    #biến câu hỏi tự nhiên thành dữ liệu có cấu trúc để tìm kiếm chính xác hơn.   
    system_prompt = """
    Bạn là chuyên gia phân tích và tìm kiếm ẩm thực. Nhiệm vụ của bạn là:
    1. Viết lại câu hỏi để tập trung vào các món ăn (Rewritten Query).
    2. chọn ra từ khóa chính.
    3. Xác định các bộ lọc nếu có như: giá cả, địa điểm, calo.
    
    Ví dụ: "Tìm quán bún chả nào rẻ ở Hoàn Kiếm"
    Trả về: {
        "rewritten_query": "địa chỉ bán bún chả giá rẻ khu vực Hoàn Kiếm",
        "keywords": ["bún chả", "giá rẻ"],
        "filters": {"location": "Hoàn Kiếm", "price": "low"}
    }
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # chuỗi xử lý
    chain = prompt | llm_model | JsonOutputParser(pydantic_object=ProcessedQuery)
    
    try:
        processed_data = chain.invoke({"question": user_raw_query})
        return processed_data
    except Exception as e:
        # trả về câu hỏi gốc nếu lỗi
        return {"rewritten_query": user_raw_query, "keywords": [], "filters": {}}

def search(user_input,llm,vector_store):
    # Xử lý câu hỏi người dùng
    llm_model = ChatOpenAI(model_name="gpt-4o", temperature=0)
    processed_query = query_processor(user_input, llm_model)
    search_query = processed_query.rewritten_query
    metadata_filter = processed_query.filters
    try:
    # Tìm kiếm trong vector store
        retriever = vector_store.as_retriever(
        query=search_query,
        k=15,
        filter=metadata_filter if metadata_filter else None
    )
    except Exception as e:
        print(f"Lỗi: {e}")
        return []
    try:
    # rerank bằng Cohere
        cohere_api_key =os.getenv("COHERE_API_KEY")
        reranker = CohereRerank(cohere_api_key,
                             model="rerank-multilingual-v3.0", top_n=5)
        compressor_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=reranker
        )
        reranked_results = compressor_retriever.invoke(search_query)
        return reranked_results
    except Exception as ce:
        print(f"Lỗi khi rerank: {ce}")
        return []
