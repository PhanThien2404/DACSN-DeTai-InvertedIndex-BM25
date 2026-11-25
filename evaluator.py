# -*- coding: utf-8 -*-
"""
Module 4: Bộ đánh giá chất lượng truy hồi (Recall@k, MRR, so sánh xếp hạng & đo thời gian ms)
Đồ án Cơ sở ngành CNTT - Đề tài : Xây dựng chỉ mục ngược và xếp hạng TF-IDF / BM25
Sinh viên thực hiện: Phan Nguyễn Ngọc Thiện - 66133552
"""

import time
from typing import List, Dict, Any, Tuple


class Evaluator:
    """
    Lớp tự cài đặt hai độ đo chuẩn trong Truy hồi thông tin (Information Retrieval):
      - Recall@k: Tỷ lệ câu truy vấn mà tài liệu đúng nằm trong Top-k kết quả đầu tiên.
      - MRR (Mean Reciprocal Rank): Trung bình nghịch đảo thứ hạng của tài liệu đúng.
    Bổ sung Tuần 4:
      - Đo thời gian truy vấn trung bình (ms/query).
      - Thống kê số truy vấn hai thuật toán xếp hạng khác nhau (Top-1, Top-k, thứ hạng đích).
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

    @classmethod
    def measure_avg_query_time_ms(
        cls,
        ranker: Any,
        test_queries: List[Dict[str, Any]],
        top_k: int = 10,
        repeats: int = 20,
        include_preprocessing: bool = True
    ) -> float:
        """
        Đo thời gian truy vấn trung bình (ms / truy vấn) bằng time.perf_counter().
        - include_preprocessing=True: đo trọn gói (tiền xử lý câu hỏi + tra Inverted Index + xếp hạng).
        - include_preprocessing=False: đo thuần tốc độ truy vấn & xếp hạng trên token đã tiền xử lý.
        """
        if not test_queries:
            return 0.0

        if include_preprocessing:
            inputs = [q["query_text"] for q in test_queries]
        else:
            inputs = [ranker.index.preprocessor.preprocess(q["query_text"]) for q in test_queries]

        # Warm-up 1 lượt
        for inp in inputs:
            ranker.rank(inp, top_k=top_k)

        start = time.perf_counter()
        for _ in range(repeats):
            for inp in inputs:
                ranker.rank(inp, top_k=top_k)
        elapsed_sec = time.perf_counter() - start

        total_queries = len(inputs) * repeats
        return (elapsed_sec / total_queries) * 1000.0

    @classmethod
    def compare_rankers(
        cls,
        ranker_a: Any,
        ranker_b: Any,
        test_queries: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        So sánh chi tiết sự khác biệt xếp hạng giữa hai thuật toán (ví dụ: TF-IDF vs BM25):
          - diff_top1_count: Số câu truy vấn có tài liệu Top-1 khác nhau.
          - diff_topk_count: Số câu truy vấn có danh sách Top-k (theo đúng thứ tự) khác nhau.
          - diff_gt_rank_count: Số câu truy vấn mà thứ hạng tài liệu đúng (ground truth) khác nhau.
          - b_better_than_a: Số câu truy vấn ranker_b xếp tài liệu đúng cao hơn ranker_a.
          - a_better_than_b: Số câu truy vấn ranker_a xếp tài liệu đúng cao hơn ranker_b.
        """
        max_k = max(top_k, len(ranker_a.index.documents))
        diff_top1_list: List[Dict[str, Any]] = []
        diff_topk_count = 0
        diff_gt_list: List[Dict[str, Any]] = []
        b_better = 0
        a_better = 0

        for q in test_queries:
            qid = q["query_id"]
            q_text = q["query_text"]
            gt_id = int(q["ground_truth_doc_id"])

            res_a = [doc_id for doc_id, _ in ranker_a.rank(q_text, top_k=max_k)]
            res_b = [doc_id for doc_id, _ in ranker_b.rank(q_text, top_k=max_k)]

            top1_a = res_a[0] if res_a else None
            top1_b = res_b[0] if res_b else None

            if top1_a != top1_b:
                diff_top1_list.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "ground_truth_doc_id": gt_id,
                    "top1_a": top1_a,
                    "top1_b": top1_b
                })

            if res_a[:top_k] != res_b[:top_k]:
                diff_topk_count += 1

            rank_a = cls.find_rank(res_a, gt_id)
            rank_b = cls.find_rank(res_b, gt_id)

            if rank_a != rank_b:
                # Đánh giá bên nào tốt hơn (rank > 0 và nhỏ hơn là tốt hơn)
                a_wins = (rank_a > 0 and (rank_b == 0 or rank_a < rank_b))
                b_wins = (rank_b > 0 and (rank_a == 0 or rank_b < rank_a))
                if b_wins:
                    b_better += 1
                    winner = "BM25"
                elif a_wins:
                    a_better += 1
                    winner = "TF-IDF"
                else:
                    winner = "Tie"

                diff_gt_list.append({
                    "query_id": qid,
                    "query_text": q_text,
                    "ground_truth_doc_id": gt_id,
                    "rank_a": rank_a,
                    "rank_b": rank_b,
                    "top1_a": top1_a,
                    "top1_b": top1_b,
                    "winner": winner
                })

        return {
            "total_queries": len(test_queries),
            "top_k_checked": top_k,
            "diff_top1_count": len(diff_top1_list),
            "diff_topk_count": diff_topk_count,
            "diff_gt_rank_count": len(diff_gt_list),
            "b_better_than_a": b_better,
            "a_better_than_b": a_better,
            "diff_top1_details": diff_top1_list,
            "diff_gt_details": diff_gt_list
        }
