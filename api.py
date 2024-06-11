from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
import pandas as pd
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

# Khởi tạo mô hình ngôn ngữ Gemini-pro
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

def get_text_chunks(text):
    # Khởi tạo công cụ chia văn bản thành các đoạn nhỏ
    text_splitter = CharacterTextSplitter(
        separator="\n",   # Tách văn bản theo ký tự xuống dòng
        chunk_size=1000,  # Kích thước mỗi đoạn văn bản
        chunk_overlap=200, # Phần chồng lấn giữa các đoạn
        length_function=len # Hàm đo độ dài đoạn văn bản
    )
    # Chia văn bản thành các đoạn nhỏ
    chunks = text_splitter.split_text(text)
    return chunks

def generate_answer(query):
    # Đọc dữ liệu từ các file CSV
    room_df = pd.read_csv('room.csv')
    restaurant_df = pd.read_csv('restaurant.csv')

    # Tạo prompt dựa trên dữ liệu từ file CSV
    prompt = """
    Bạn là trợ lý khách sạn, nhiệm vụ của bạn là hỗ trợ khách hàng tìm hiểu và đặt phòng khách sạn cũng như thông tin về nhà hàng trong khách sạn dựa trên dữ liệu từ các tệp CSV được cung cấp. Mục tiêu của bạn là cung cấp câu trả lời chính xác và thân thiện dựa trên thông tin có trong tệp dữ liệu. Nếu không thể tìm thấy câu trả lời trong dữ liệu có sẵn, hãy thông báo rõ ràng điều đó cho khách hàng.

    ### Thông tin khách sạn
    - Tên: Nha Trang Hotel (Khách sạn Nha Trang).
    - Địa chỉ: Số 3 Quang Trung, Nha Trang.
    - Mô tả: Tự hào là khách sạn 5 sao hàng đầu tại Nha Trang, chúng tôi luôn mang đến cho quý khách những trải nghiệm tuyệt vời nhất.
    - Email: NThotel@gmail.com
    - SĐT: + 01 234 567 88

    ### Thông tin về phòng khách sạn:
    """

    # Thêm thông tin về các loại phòng từ file CSV
    for index, row in room_df.iterrows():
        prompt += f"""
    1. **{row['room_type']}**
    - Giá cho 1 đêm: {row['room_price']}
    - Số người tối đa: {row['max_people']}
    - Mô tả: {row['description']}
    """

    prompt += "\n### Thông tin về nhà hàng trong khách sạn:\n"

    # Thêm thông tin về các nhà hàng từ file CSV
    for index, row in restaurant_df.iterrows():
        prompt += f"""
    1. **{row['name']}**
    - Địa điểm: {row['location']}
    - Số điện thoại: {row['phone_number']}
    - Giờ hoạt động: {row['hours']}
    - Email: {row['email']}
    - Mô tả: {row['description']}
    """

    prompt += """\nLưu ý: Nếu dữ liệu là tiếng Anh, bạn có thể dịch sang tiếng Việt để hiểu rõ hơn.
    """

    # Khởi tạo mô hình nhúng văn bản từ Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Lấy văn bản gốc từ thông tin khách sạn
    raw_text = prompt
    # Chia văn bản thành các đoạn nhỏ
    text_chunks = get_text_chunks(raw_text)

    # Tạo kho lưu trữ vector từ các đoạn văn bản và nhúng văn bản
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # Tạo bộ truy vấn từ kho vector, tìm kiếm 3 kết quả gần nhất
    db = vectorstore.as_retriever(search_kwargs={'k': 3})

    # Tạo mẫu prompt để trả lời câu hỏi dựa trên dữ liệu và lịch sử hội thoại
    template_qah = "Dựa vào dữ liệu sau để trả lời câu hỏi\n{context}\n{chat_history}\n### Câu hỏi:\n{question}\n\n### Trả lời:"
    prompt_qah = PromptTemplate(template=template_qah, input_variables=["question"])

    # Tạo mẫu prompt chỉ sử dụng lịch sử hội thoại và câu hỏi
    template_qah_1 = "Lịch sử:\n{chat_history}\n### Câu hỏi:\n{question}\n\n### Trả lời:"
    prompt_qah_1 = PromptTemplate(template=template_qah_1, input_variables=["question"])

    # Thiết lập bộ nhớ hội thoại để lưu trữ lịch sử hội thoại
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Tạo chuỗi truy vấn hội thoại với bộ nhớ, sử dụng mô hình ngôn ngữ và bộ truy vấn
    qah_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db,
        return_source_documents=False,
        verbose=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt_qah},
        condense_question_prompt=prompt_qah_1
    )

    # Thực hiện truy vấn và lấy kết quả trả lời
    sol = qah_chain({"question": query})

    return sol['answer']

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # Lấy dữ liệu JSON từ yêu cầu POST
    question = data.get('question')  # Lấy câu hỏi từ dữ liệu JSON
    
    if not question:  # Kiểm tra nếu không có câu hỏi
        return jsonify({'error': 'No question provided'}), 400  # Trả về lỗi 400
    
    result = generate_answer(question)  # Tạo câu trả lời cho câu hỏi
    return jsonify({'answer': result})  # Trả về câu trả lời dưới dạng JSON

if __name__ == '__main__':
    app.run(debug=True, port=8080, use_reloader=False)  # Chạy ứng dụng Flask
