# -*- coding: utf-8 -*-
"""
Module 3: Bộ xếp hạng TF-IDF và Okapi BM25 (TFIDFRanker & BM25Ranker)
Đồ án Cơ sở ngành CNTT - Đề tài : Xây dựng chỉ mục ngược và xếp hạng TF-IDF / BM25
Sinh viên thực hiện: Phan Nguyễn Ngọc Thiện - 66133552
"""

import math
import heapq
from typing import Dict, List, Tuple
from inverted_index import InvertedIndex


class TFIDFRanker:
    """
    Bộ xếp hạng TF-IDF tự cài đặt theo đúng công thức Mục 2.3 trong Báo cáo:
      - TF(t, D) = 1 + ln(f_{t,D}) khi f_{t,D} > 0, ngược lại = 0
      - IDF(t)   = ln((N + 1) / (df_t + 1)) + 1
      - Score(Q, D) = sum_{t in Q \cap D} TF(t, D) * IDF(t)
    """

    def __init__(self, index: InvertedIndex):
        self.index: InvertedIndex = index
        self.idf_cache: Dict[str, float] = {}
        self._precompute_idf()

    def _precompute_idf(self) -> None:
        """Tính sẵn trọng số IDF(t) cho toàn bộ từ vựng trong O(V)."""
        N = self.index.total_docs
        for term, df_t in self.index.vocab.items():
            self.idf_cache[term] = math.log((N + 1.0) / (df_t + 1.0)) + 1.0

    def compute_idf(self, term: str) -> float:
        """Trả về trọng số IDF(t) đã tính sẵn."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        N = self.index.total_docs
        df_t = self.index.get_df(term)
        return math.log((N + 1.0) / (df_t + 1.0)) + 1.0

    @staticmethod
    def compute_tf(tf_val: int) -> float:
        """Tính trọng số TF chuẩn hóa logarit: 1 + ln(f_{t,D})."""
        if tf_val <= 0:
            return 0.0
        return 1.0 + math.log(tf_val)

    def score_all(self, query_Text_or_tokens: str | List[str]) -> Dict[int, float]:
        """
        Tính điểm TF-IDF cho các tài liệu ứng viên chứa ít nhất 1 từ khóa trong truy vấn Q.
        Độ phức tạp: O(|Q| * P_avg).
        """
        if isinstance(query_Text_or_tokens, str):
            q_tokens = self.index.preprocessor.preprocess(query_Text_or_tokens)
        else:
            q_tokens = query_Text_or_tokens

        # Giữ thứ tự từ khóa duy nhất trong câu hỏi Q
        unique_q_terms = list(dict.fromkeys(q_tokens))
        scores: Dict[int, float] = {}

        for term in unique_q_terms:
            if term not in self.index.vocab:
                continue
            idf_val = self.compute_idf(term)
            for doc_id, tf_val in self.index.get_postings(term):
                tf_weight = self.compute_tf(tf_val)
                scores[doc_id] = scores.get(doc_id, 0.0) + (tf_weight * idf_val)

        return scores

    def rank(self, query: str | List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Trích xuất Top-k tài liệu có điểm TF-IDF cao nhất bằng cấu trúc Min-Heap (heapq).
        Độ phức tạp: O(|Q| * P_avg + C log k).
        """
        scores = self.score_all(query)
        if not scores:
            return []
        # Sử dụng heapq.nlargest (duy trì Min-Heap kích thước top_k nội bộ)
        top_results = heapq.nlargest(
            top_k,
            scores.items(),
            key=lambda item: (item[1], -item[0])
        )
        return top_results


class BM25Ranker:
    """
    Bộ xếp hạng xác suất Okapi BM25 tự cài đặt theo đúng công thức Mục 2.4 trong Báo cáo:
      - IDF_BM25(t) = ln(1 + (N - df_t + 0.5) / (df_t + 0.5))
      - K_D = k1 * (1 - b + b * (|D| / avgdl))
      - Score_BM25(Q, D) = sum_{t in Q \cap D} IDF_BM25(t) * [f_{t,D} * (k1 + 1)] / (f_{t,D} + K_D)
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index: InvertedIndex = index
        self.k1: float = k1
        self.b: float = b
        self.idf_cache: Dict[str, float] = {}
        self.kd_cache: Dict[int, float] = {}
        self._precompute_constants()

    def _precompute_constants(self) -> None:
        """Tính sẵn trọng số IDF_BM25(t) và hệ số chuẩn hóa độ dài K_D cho mọi tài liệu."""
        N = self.index.total_docs
        avgdl = self.index.avgdl if self.index.avgdl > 0 else 1.0

        for term, df_t in self.index.vocab.items():
            self.idf_cache[term] = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))

        for doc_id, doc_len in self.index.doc_lengths.items():
            self.kd_cache[doc_id] = self.k1 * (
                1.0 - self.b + self.b * (doc_len / avgdl)
            )

    def compute_idf(self, term: str) -> float:
        """Trả về trọng số IDF_BM25(t)."""
        if term in self.idf_cache:
            return self.idf_cache[term]
        N = self.index.total_docs
        df_t = self.index.get_df(term)
        return math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))

    def compute_kd(self, doc_id: int) -> float:
        """Trả về hệ số chuẩn hóa độ dài tài liệu K_D."""
        return self.kd_cache[doc_id]

    def score_all(self, query_text_or_tokens: str | List[str]) -> Dict[int, float]:
        """
        Tính điểm Okapi BM25 cho các tài liệu ứng viên trên Chỉ mục ngược.
        Độ phức tạp: O(|Q| * P_avg).
        """
        if isinstance(query_text_or_tokens, str):
            q_tokens = self.index.preprocessor.preprocess(query_text_or_tokens)
        else:
            q_tokens = query_text_or_tokens

        unique_q_terms = list(dict.fromkeys(q_tokens))
        scores: Dict[int, float] = {}

        for term in unique_q_terms:
            if term not in self.index.vocab:
                continue
            idf_val = self.compute_idf(term)
            for doc_id, tf_val in self.index.get_postings(term):
                kd_val = self.kd_cache[doc_id]
                numerator = tf_val * (self.k1 + 1.0)
                denominator = tf_val + kd_val
                term_score = idf_val * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        return scores

    def rank(self, query: str | List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Trích xuất Top-k tài liệu có điểm BM25 cao nhất bằng Min-Heap (heapq).
        Độ phức tạp: O(|Q| * P_avg + C log k).
        """
        scores = self.score_all(query)
        if not scores:
            return []
        top_results = heapq.nlargest(
            top_k,
            scores.items(),
            key=lambda item: (item[1], -item[0])
        )
        return top_results
