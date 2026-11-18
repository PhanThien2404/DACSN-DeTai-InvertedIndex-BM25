# -*- coding: utf-8 -*-
"""
Module 2: Cấu trúc dữ liệu Chỉ mục ngược (Inverted Index)
Đồ án Cơ sở ngành CNTT - Đề tài : Xây dựng chỉ mục ngược và xếp hạng TF-IDF / BM25
Sinh viên thực hiện: Phan Nguyễn Ngọc Thiện - 66133552
"""

from typing import Dict, List, Tuple, Any, Optional
from preprocessor import VietnamesePreprocessor


class InvertedIndex:
    """
    Cấu trúc dữ liệu Chỉ mục ngược (Inverted Index) tự cài đặt từ đầu bằng bảng băm (dict).
    Lưu trữ:
      - vocab (Dict[str, int]): Từ điển lưu tần suất tài liệu df_t của mỗi từ t.
      - postings (Dict[str, List[Tuple[int, int]]]): Danh sách bài đăng [(doc_id, tf_{t,D})].
      - doc_lengths (Dict[int, int]): Độ dài |D| (tổng số từ sau tiền xử lý) của từng tài liệu.
      - avgdl (float): Độ dài trung bình của tài liệu trên toàn kho dữ liệu.
      - total_docs (int): Tổng số tài liệu N.
    """

    def __init__(
        self,
        preprocessor: Optional[VietnamesePreprocessor] = None,
        index_field: str = "both"
    ):
        if index_field not in ("both", "question", "answer"):
            raise ValueError("index_field phải là 'both', 'question' hoặc 'answer'.")
        self.preprocessor: VietnamesePreprocessor = (
            preprocessor if preprocessor is not None else VietnamesePreprocessor()
        )
        self.index_field: str = index_field

        self.vocab: Dict[str, int] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.doc_tokens: Dict[int, List[str]] = {}
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.total_docs: int = 0
        self.avgdl: float = 0.0

    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """Trích xuất văn bản để đánh chỉ mục theo cấu hình index_field."""
        if "text" in doc and "question" not in doc:
            return str(doc["text"])
        q_text = str(doc.get("question", "")).strip()
        a_text = str(doc.get("answer", "")).strip()
        if self.index_field == "question":
            return q_text
        elif self.index_field == "answer":
            return a_text
        return f"{q_text} {a_text}".strip()

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        """
        Xây dựng Chỉ mục ngược từ danh sách tài liệu.
        Độ phức tạp thời gian: O(N * L_avg) = O(T).
        """
        self.vocab.clear()
        self.postings.clear()
        self.doc_lengths.clear()
        self.doc_tokens.clear()
        self.documents.clear()

        total_length = 0

        for doc in corpus:
            doc_id = int(doc["doc_id"])
            self.documents[doc_id] = doc
            raw_text = self._extract_text(doc)
            tokens = self.preprocessor.preprocess(raw_text)

            self.doc_tokens[doc_id] = tokens
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_length += doc_len

            # Đếm tần suất từ tf_{t,D} trong tài liệu hiện tại bằng bảng băm cục bộ
            term_counts: Dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            # Cập nhật vào Vocabulary (df_t) và Posting List [(doc_id, tf)]
            for term, tf_val in term_counts.items():
                if term not in self.postings:
                    self.postings[term] = []
                    self.vocab[term] = 0
                self.postings[term].append((doc_id, tf_val))
                self.vocab[term] += 1

        self.total_docs = len(corpus)
        self.avgdl = (total_length / self.total_docs) if self.total_docs > 0 else 0.0

    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        """Tra cứu Posting List của từ khóa t trong thời gian O(1)."""
        return self.postings.get(term, [])

    def get_df(self, term: str) -> int:
        """Lấy số lượng tài liệu chứa từ t (df_t) trong thời gian O(1)."""
        return self.vocab.get(term, 0)
