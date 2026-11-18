# -*- coding: utf-8 -*-
"""
Module 4: Bộ đánh giá chất lượng truy hồi (Recall@k và MRR)
Đồ án Cơ sở ngành CNTT - Đề tài : Xây dựng chỉ mục ngược và xếp hạng TF-IDF / BM25
Sinh viên thực hiện: Phan Nguyễn Ngọc Thiện - 66133552
"""

from typing import List, Dict, Any, Tuple


class Evaluator:
    """
    Lớp tự cài đặt hai độ đo chuẩn trong Truy hồi thông tin (Information Retrieval):
      - Recall@k: Tỷ lệ câu truy vấn mà tài liệu đúng nằm trong Top-k kết quả đầu tiên.
      - MRR (Mean Reciprocal Rank): Trung bình nghịch đảo thứ hạng của tài liệu đúng.
    """

    @staticmethod
    def find_rank(ranked_doc_ids: List[int], ground_truth_doc_id: int) -> int:
        """
        Tìm thứ hạng (1-based index) của tài liệu đúng trong danh sách kết quả.
        Trả về 0 nếu không tìm thấy.
        """
        for idx, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id == ground_truth_doc_id:
                return idx
        return 0

    @staticmethod
    def recall_at_k(ranks: List[int], k: int) -> float:
        """
        Tính Recall@k từ danh sách thứ hạng rank_i của từng câu hỏi.
        Công thức: (1 / |Q|) * sum_{i=1}^{|Q|} I(1 <= rank_i <= k)
        """
        if not ranks:
            return 0.0
        hits = sum(1 for r in ranks if 1 <= r <= k)
        return hits / len(ranks)

    @staticmethod
    def mrr(ranks: List[int]) -> float:
        """
        Tính Mean Reciprocal Rank (MRR) từ danh sách thứ hạng rank_i.
        Công thức: (1 / |Q|) * sum_{i=1}^{|Q|} (1 / rank_i) (nếu rank_i > 0)
        """
        if not ranks:
            return 0.0
        reciprocal_sum = sum((1.0 / r) if r > 0 else 0.0 for r in ranks)
        return reciprocal_sum / len(ranks)

    @classmethod
    def evaluate_ranker(
        cls,
        ranker: Any,
        test_queries: List[Dict[str, Any]],
        k_values: Tuple[int, ...] = (1, 3, 5, 10)
    ) -> Dict[str, float]:
        """Đánh giá toàn diện một bộ xếp hạng trên tập test_queries."""
        max_k = max(max(k_values), len(ranker.index.documents))
        ranks: List[int] = []

        for q in test_queries:
            q_text = q["query_text"]
            gt_id = int(q["ground_truth_doc_id"])
            ranked_pairs = ranker.rank(q_text, top_k=max_k)
            ranked_ids = [doc_id for doc_id, _ in ranked_pairs]
            r = cls.find_rank(ranked_ids, gt_id)
            ranks.append(r)

        metrics: Dict[str, float] = {}
        for k in k_values:
            metrics[f"Recall@{k}"] = cls.recall_at_k(ranks, k)
        metrics["MRR"] = cls.mrr(ranks)
        return metrics
