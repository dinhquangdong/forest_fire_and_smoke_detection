Các thành viên nhóm:
- Hồ Hồng Hà - 20520480
- Trương Văn Chinh - 20521137
- Đinh Quang Đông - 20521189

Trong thư mục source có folder training là các file .ipynb có chứa các kết quả của quá trình train model.

Hướng dẫn build và thực thi code:
    1. Mở command prompt hoặc terminal
    2. Nếu dùng anaconda, chạy lần lượt các câu lệnh sau để tạo môi trường ảo:
    conda create --name fire_and_smoke_detection
    conda activate fire_and_smoke_detection

    3. Dùng câu lệnh sau để di chuyển vào thư mục source:
    cd source

    4. Chạy câu lệnh sau để cài đặt các thư viện cần thiết:
    pip install -r requirements.txt

    5. Chạy câu lệnh sau để chạy demo của chương trình:
    streamlit run demo.py

Link dataset: https://universe.roboflow.com/kirzone/fire-iejes/dataset/2
