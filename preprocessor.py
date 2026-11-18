# -*- coding: utf-8 -*-
"""
Module 1: Tiền xử lý văn bản tiếng Việt (Vietnamese Text Preprocessor)
Đồ án Cơ sở ngành CNTT - Đề tài : Xây dựng chỉ mục ngược và xếp hạng TF-IDF / BM25
Sinh viên thực hiện: Phan Nguyễn Ngọc Thiện - 66133552
"""

import re
from typing import List, Set, Optional

# Thử import thư viện tách từ tiếng Việt (underthesea hoặc pyvi) nếu đã cài đặt
try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

try:
    from underthesea import word_tokenize as uts_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False


# Danh sách từ dừng (Stopwords) tiếng Việt phổ biến trong miền dữ liệu FAQ tuyển sinh
DEFAULT_STOPWORDS: Set[str] = {
    "và", "cho", "hệ", "diện", "là", "của", "những", "các", "thì", "mà", "tại", "ở",
    "với", "trong", "được", "có", "này", "đó", "như", "về", "theo", "từ", "đến",
    "khi", "nếu", "sẽ", "đã", "đang", "vào", "ra", "lên", "xuống", "hay", "hoặc",
    "nhưng", "vì", "do", "bởi", "để", "cũng", "rất", "quá", "lại", "nữa", "vậy",
    "ạ", "nhé", "nha", "vâng", "dạ", "ơi", "à", "ừ", "hả", "sao", "nào", "gì",
    "ai", "đâu", "bao_nhiêu", "thế_nào", "như_thế_nào", "ra_sao"
}

# Từ điển từ ghép chuyên ngành Tuyển sinh & CNTT đảm bảo tách từ chuẩn xác
# ngay cả khi chạy độc lập không cần thư viện ngoài (và khớp 100% ví dụ tính tay Chương 3)
DOMAIN_COMPOUND_WORDS: List[str] = [
    "công nghệ thông tin", "kỹ thuật phần mềm", "công nghệ phần mềm", "hệ thống thông tin",
    "mạng máy tính", "trí tuệ nhân tạo", "khoa học dữ liệu", "đại học nha trang",
    "đánh giá năng lực", "tốt nghiệp thpt", "xét tuyển thẳng", "ngân hàng chính sách xã hội",
    "khuyến khích học tập", "nghĩa vụ quân sự", "cơ sở vật chất", "đồ án cơ sở ngành",
    "đồ án tốt nghiệp", "giáo dục thể chất", "giáo dục quốc phòng", "căn cước công dân",
    "giấy khai sinh", "bảo hiểm y tế", "khu vực", "đối tượng", "nguyện vọng",
    "học phí", "điểm chuẩn", "trúng tuyển", "đại trà", "quy định", "chính sách",
    "miễn giảm", "sinh viên", "tân sinh viên", "tín chỉ", "học bạ", "tổ hợp",
    "chuẩn đầu ra", "tiếng anh", "ngoại ngữ", "học bổng", "thủ khoa", "ký túc xá",
    "nội trú", "đồi la san", "nhập học", "xác nhận", "trực tuyến", "thực tập",
    "doanh nghiệp", "việc làm", "phương thức", "xét tuyển", "tuyển sinh", "mã trường",
    "điểm sàn", "chất lượng cao", "song ngành", "học vượt", "học lại", "cải thiện",
    "điểm rèn luyện", "thư viện", "phòng máy"
]


class VietnamesePreprocessor:
    """
    Lớp tiền xử lý văn bản tiếng Việt thực hiện 3 bước:
    1. Chuẩn hóa văn bản (Normalization: lowercase, xóa dấu câu & ký tự đặc biệt).
    2. Tách từ tiếng Việt (Word Segmentation: ghép âm tiết bằng dấu '_').
    3. Loại bỏ từ dừng (Stopwords Removal).
    """

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        compound_words: Optional[List[str]] = None,
        use_external_tokenizer: bool = True
    ):
        self.stopwords: Set[str] = stopwords if stopwords is not None else set(DEFAULT_STOPWORDS)
        # Sắp xếp từ ghép dài nhất lên trước (Longest Matching) để ưu tiên cụm dài
        raw_compounds = compound_words if compound_words is not None else DOMAIN_COMPOUND_WORDS
        self.compound_words: List[str] = sorted(raw_compounds, key=lambda x: len(x.split()), reverse=True)
        self.use_external_tokenizer: bool = use_external_tokenizer

    def normalize(self, text: str) -> str:
        """Bước 1: Chuyển chữ thường, loại bỏ dấu câu và chuẩn hóa khoảng trắng."""
        if not text:
            return ""
        text = text.lower()
        # Giữ lại chữ cái tiếng Việt có dấu, chữ số và dấu gạch dưới
        text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def segment_words(self, text: str) -> str:
        """Bước 2: Tách từ tiếng Việt (ghép các từ đa âm tiết bằng dấu '_')."""
        if not text:
            return ""

        # Ưu tiên khớp các thuật ngữ từ ghép miền tuyển sinh & Chương 3 trước (Longest Matching)
        segmented = f" {text} "
        for phrase in self.compound_words:
            connected = phrase.replace(" ", "_")
            pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
            segmented = re.sub(pattern, connected, segmented)
        segmented = segmented.strip()

        # Nếu bật cờ dùng thư viện ngoài (pyvi) cho các từ còn lại chưa nối
        if self.use_external_tokenizer and HAS_PYVI:
            # Bảo vệ các cụm đã nối sẵn dấu '_'
            tokens = ViTokenizer.tokenize(segmented).split()
            return " ".join(tokens)

        return segmented

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Bước 3: Lọc bỏ các từ dừng (Stopwords)."""
        return [t for t in tokens if t and t not in self.stopwords]

    def preprocess(self, text: str) -> List[str]:
        """Thực thi trọn vẹn pipeline tiền xử lý văn bản và trả về danh sách tokens."""
        norm_text = self.normalize(text)
        seg_text = self.segment_words(norm_text)
        tokens = seg_text.split()
        clean_tokens = self.remove_stopwords(tokens)
        return clean_tokens
