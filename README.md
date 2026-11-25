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
├── evaluator.py                # Module 4: Độ đo Recall@k, MRR, so sánh khác biệt xếp hạng & đo thời gian truy vấn (ms)
├── baseline_compare.py         # Module 5: Đối chứng kết quả với thư viện chuẩn scikit-learn (TfidfVectorizer)
├── test_hand_calc.py           # Unit Test tái hiện chính xác 100% ví dụ tính tay Chương 3 (Tuần 3)
├── experiments_tuan4.py        # Thực nghiệm Tuần 4: Khảo sát index_field, lưới (k1, b), khác biệt xếp hạng & tốc độ (ms)
├── faq_data_Tuan3.json         # Bộ dữ liệu Tuần 3 (60 tài liệu gốc & 120 truy vấn)
├── test_queries_Tuan3.json     # Tập 120 câu hỏi kiểm thử (60 từ Bạn học + 60 từ GenAI)
├── faq_data_Tuan4.json         # Bộ dữ liệu Tuần 4 (60 gốc + 30 tài liệu gây nhiễu cùng chủ đề = 90 Docs, có URL cụ thể)
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

## 3. Hướng dẫn chạy kiểm thử & thực nghiệm

### 3.1. Chạy Unit Test tái hiện Ví dụ tính tay Chương 3 (Tuần 3)
```bash
python test_hand_calc.py
```

### 3.2. Chạy đối chứng `TFIDFRanker` tự cài vs. `scikit-learn` (`TfidfVectorizer`)
```bash
python baseline_compare.py
```

### 3.3. Chạy thực nghiệm Tuần 4 (90 tài liệu & 120 truy vấn)
Khảo sát `index_field` (`question` / `answer` / `both`), lưới tham số BM25 ($k_1 \in \{0.9, 1.2, 1.5, 2.0\}$, $b \in \{0.25, 0.5, 0.75, 1.0\}$), đếm số truy vấn xếp hạng khác nhau và đo thời gian truy vấn trung bình (ms):
```bash
python experiments_tuan4.py
```
