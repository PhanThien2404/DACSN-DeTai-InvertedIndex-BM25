# Đồ Án Cơ Sở Ngành CNTT - Đề tài : Xây dựng Chỉ mục ngược và Xếp hạng TF-IDF / BM25

- **Trường:** Đại học Nha Trang (Nha Trang University - NTU) - Khoa Công nghệ Thông tin
- **Giảng viên hướng dẫn:** Thầy Ngô Nguyễn Tường Nghi
- **Sinh viên thực hiện:** Phan Nguyễn Ngọc Thiện – MSSV: `66133552` (Lớp `66CNPM`)

---

## 1. Cấu trúc thư mục dự án

```text
DACSN_DeTai_InvertedIndex_BM25/
├── preprocessor.py             # Module 1: Tiền xử lý văn bản tiếng Việt (Normalization, Word Segmentation, Stopwords)
├── inverted_index.py           # Module 2: Cấu trúc dữ liệu Chỉ mục ngược (InvertedIndex tự cài bằng bảng băm)
├── ranker.py                   # Module 3: Bộ xếp hạng TFIDFRanker và BM25Ranker (tự cài công thức toán học + Min-Heap)
├── evaluator.py                # Module 4: Độ đo đánh giá chất lượng truy hồi Recall@k và MRR
├── baseline_compare.py         # Module 5: Đối chứng kết quả với thư viện chuẩn scikit-learn (TfidfVectorizer)
├── test_hand_calc.py           # Unit Test tái hiện chính xác 100% ví dụ tính tay Chương 3 & chạy thử nghiệm 60 Docs
├── faq_data_Tuan3.json         # Kho dữ liệu 60 tài liệu FAQ Tuyển sinh ĐH Nha Trang (kèm nguồn chính thức)
├── test_queries_Tuan3.json     # Tập 120 câu hỏi kiểm thử (mỗi tài liệu 2 truy vấn từ Bạn học & GenAI)
└── README.md                   # Tài liệu hướng dẫn cài đặt và chạy thực nghiệm
```

---

## 2. Hướng dẫn cài đặt môi trường Python

Yêu cầu **Python 3.9 trở lên**. Mở Terminal (hoặc PowerShell trên VS Code) tại thư mục dự án và cài đặt các thư viện hỗ trợ:

```bash
pip install pyvi scikit-learn rank_bm25 pandas openpyxl
```

_(Lưu ý: Toàn bộ lõi thuật toán `InvertedIndex`, `TFIDFRanker`, `BM25Ranker`, `Evaluator` đều được tự cài đặt bằng Python thuần và `math`/`heapq` nên có thể chạy ngay cả khi chưa cài thư viện ngoài)._

---

## 3. Cách chạy Unit Test tái hiện Ví dụ tính tay Chương 3 & Thực nghiệm Tuần 3

Chạy lệnh sau trên Terminal để kiểm tra toàn bộ 5 bài Unit Test tự động và xem bảng kết quả trên 60 tài liệu FAQ (120 câu truy vấn):

```bash
python test_hand_calc.py
```

Để chạy đối chứng độ khớp điểm số tuyệt đối giữa `TFIDFRanker` tự cài và `TfidfVectorizer` của `scikit-learn`:

```bash
python baseline_compare.py
```
