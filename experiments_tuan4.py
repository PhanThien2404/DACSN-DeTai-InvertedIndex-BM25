# -*- coding: utf-8 -*-
"""
Thực nghiệm Tuần 4:
  1. Khảo sát trường đánh chỉ mục index_field ('question', 'answer', 'both')
     cho cả TF-IDF và BM25 trên bộ dữ liệu 90 tài liệu (60 gốc + 30 gây nhiễu) & 120 truy vấn.
  2. Khảo sát lưới tham số BM25: k1 in {0.9, 1.2, 1.5, 2.0} x b in {0.25, 0.5, 0.75, 1.0}.
  3. Báo cáo số truy vấn hai thuật toán xếp khác nhau (Top-1, Top-5, thứ hạng tài liệu đích)
     và đo thời gian truy vấn trung bình (ms).

Đồ án Cơ sở ngành CNTT - Đề tài : Xây dựng chỉ mục ngược và xếp hạng TF-IDF / BM25
Sinh viên thực hiện: Phan Nguyễn Ngọc Thiện - 66133552
"""

import json
import os
from typing import Dict, Any, List, Tuple
from preprocessor import VietnamesePreprocessor
from inverted_index import InvertedIndex
from ranker import TFIDFRanker, BM25Ranker
from evaluator import Evaluator


DATA_FILE = "faq_data_Tuan4.json"
K1_GRID = [0.9, 1.2, 1.5, 2.0]
B_GRID = [0.25, 0.5, 0.75, 1.0]
INDEX_FIELDS = ["question", "answer", "both"]


def load_dataset(filepath: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu '{filepath}'. "
            "Hãy đảm bảo file faq_data_Tuan4.json nằm cùng thư mục với mã nguồn."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["corpus"], data["test_queries"]


def run_index_field_experiment(
    corpus: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    repeats: int = 30
) -> List[Dict[str, Any]]:
    """Thí nghiệm 1: So sánh index_field = question / answer / both cho TF-IDF và BM25."""
    print("=" * 95)
    print("THÍ NGHIỆM 1: KHẢO SÁT TRƯỜNG ĐÁNH CHỈ MỤC (index_field: question / answer / both)")
    print("=" * 95)

    results: List[Dict[str, Any]] = []
    header = (
        f"{'index_field':<11} | {'Thuật toán':<10} | {'V':>4} | {'avgdl':>6} | "
        f"{'R@1':>7} | {'R@3':>7} | {'R@5':>7} | {'R@10':>7} | {'MRR':>7} | {'ms(full)':>8} | {'ms(rank)':>8}"
    )
    print(header)
    print("-" * len(header))

    for field in INDEX_FIELDS:
        prep = VietnamesePreprocessor()
        idx = InvertedIndex(preprocessor=prep, index_field=field)
        idx.build_index(corpus)

        tfidf = TFIDFRanker(idx)
        bm25 = BM25Ranker(idx, k1=1.5, b=0.75)

        for algo_name, ranker in [("TF-IDF", tfidf), ("BM25", bm25)]:
            metrics = Evaluator.evaluate_ranker(ranker, queries, k_values=(1, 3, 5, 10))
            ms_full = Evaluator.measure_avg_query_time_ms(
                ranker, queries, top_k=10, repeats=repeats, include_preprocessing=True
            )
            ms_rank = Evaluator.measure_avg_query_time_ms(
                ranker, queries, top_k=10, repeats=repeats * 2, include_preprocessing=False
            )

            row = {
                "index_field": field,
                "algorithm": algo_name,
                "vocab_size": len(idx.vocab),
                "avgdl": round(idx.avgdl, 2),
                "Recall@1": round(metrics["Recall@1"] * 100, 2),
                "Recall@3": round(metrics["Recall@3"] * 100, 2),
                "Recall@5": round(metrics["Recall@5"] * 100, 2),
                "Recall@10": round(metrics["Recall@10"] * 100, 2),
                "MRR": round(metrics["MRR"], 4),
                "avg_ms_full": round(ms_full, 4),
                "avg_ms_rank_only": round(ms_rank, 4)
            }
            results.append(row)
            print(
                f"{field:<11} | {algo_name:<10} | {row['vocab_size']:>4} | {row['avgdl']:>6.2f} | "
                f"{row['Recall@1']:>6.2f}% | {row['Recall@3']:>6.2f}% | {row['Recall@5']:>6.2f}% | "
                f"{row['Recall@10']:>6.2f}% | {row['MRR']:>7.4f} | {row['avg_ms_full']:>8.4f} | {row['avg_ms_rank_only']:>8.4f}"
            )
    print("-" * len(header))
    return results


def run_bm25_grid_search(
    corpus: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    index_field: str = "both"
) -> Dict[str, Any]:
    """Thí nghiệm 2: Khảo sát lưới tham số k1 in {0.9, 1.2, 1.5, 2.0} x b in {0.25, 0.5, 0.75, 1.0}."""
    print("\n" + "=" * 95)
    print(f"THÍ NGHIỆM 2: KHẢO SÁT LƯỚI THAM SỐ BM25 (k1 x b) TRÊN index_field='{index_field}'")
    print("=" * 95)

    prep = VietnamesePreprocessor()
    idx = InvertedIndex(preprocessor=prep, index_field=index_field)
    idx.build_index(corpus)

    grid_rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] = {}

    header = f"{'k1':>5} | {'b':>5} | {'Recall@1':>9} | {'Recall@3':>9} | {'Recall@5':>9} | {'Recall@10':>9} | {'MRR':>8}"
    print(header)
    print("-" * len(header))

    for k1 in K1_GRID:
        for b in B_GRID:
            bm25 = BM25Ranker(idx, k1=k1, b=b)
            metrics = Evaluator.evaluate_ranker(bm25, queries, k_values=(1, 3, 5, 10))
            row = {
                "k1": k1,
                "b": b,
                "Recall@1": round(metrics["Recall@1"] * 100, 2),
                "Recall@3": round(metrics["Recall@3"] * 100, 2),
                "Recall@5": round(metrics["Recall@5"] * 100, 2),
                "Recall@10": round(metrics["Recall@10"] * 100, 2),
                "MRR": round(metrics["MRR"], 4)
            }
            grid_rows.append(row)
            if not best_row or (row["MRR"], row["Recall@1"]) > (best_row["MRR"], best_row["Recall@1"]):
                best_row = row

            print(
                f"{k1:>5.1f} | {b:>5.2f} | {row['Recall@1']:>8.2f}% | {row['Recall@3']:>8.2f}% | "
                f"{row['Recall@5']:>8.2f}% | {row['Recall@10']:>8.2f}% | {row['MRR']:>8.4f}"
            )

    print("-" * len(header))
    print(
        f"==> Cấu hình tối ưu trên lưới: k1 = {best_row['k1']}, b = {best_row['b']} "
        f"(Recall@1 = {best_row['Recall@1']:.2f}%, Recall@5 = {best_row['Recall@5']:.2f}%, MRR = {best_row['MRR']:.4f})"
    )
    return {"index_field": index_field, "grid": grid_rows, "best": best_row}


def run_ranking_diff_and_timing(
    corpus: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    index_field: str = "both",
    best_k1: float = 1.5,
    best_b: float = 0.75,
    repeats: int = 50
) -> Dict[str, Any]:
    """Thí nghiệm 3: Đếm số truy vấn hai thuật toán xếp khác nhau & đo thời gian trung bình (ms)."""
    print("\n" + "=" * 95)
    print(f"THÍ NGHIỆM 3: SO SÁNH KHÁC BIỆT XẾP HẠNG & THỜI GIAN TRUY VẤN (TF-IDF vs BM25 k1={best_k1}, b={best_b})")
    print("=" * 95)

    prep = VietnamesePreprocessor()
    idx = InvertedIndex(preprocessor=prep, index_field=index_field)
    idx.build_index(corpus)

    tfidf = TFIDFRanker(idx)
    bm25 = BM25Ranker(idx, k1=best_k1, b=best_b)

    diff = Evaluator.compare_rankers(tfidf, bm25, queries, top_k=5)

    # Đo thời gian trung bình lặp lại nhiều lần để ổn định
    tfidf_ms_full = Evaluator.measure_avg_query_time_ms(tfidf, queries, top_k=10, repeats=repeats, include_preprocessing=True)
    bm25_ms_full = Evaluator.measure_avg_query_time_ms(bm25, queries, top_k=10, repeats=repeats, include_preprocessing=True)

    tfidf_ms_rank = Evaluator.measure_avg_query_time_ms(tfidf, queries, top_k=10, repeats=repeats * 2, include_preprocessing=False)
    bm25_ms_rank = Evaluator.measure_avg_query_time_ms(bm25, queries, top_k=10, repeats=repeats * 2, include_preprocessing=False)

    print(f"- Tổng số câu truy vấn kiểm thử: {diff['total_queries']}")
    print(f"- Số truy vấn khác tài liệu Top-1 giữa TF-IDF và BM25: {diff['diff_top1_count']} / {diff['total_queries']} ({diff['diff_top1_count']/diff['total_queries']*100:.2f}%)")
    print(f"- Số truy vấn khác danh sách thứ tự Top-5 giữa TF-IDF và BM25: {diff['diff_topk_count']} / {diff['total_queries']} ({diff['diff_topk_count']/diff['total_queries']*100:.2f}%)")
    print(f"- Số truy vấn có thứ hạng tài liệu đích (ground truth) khác nhau: {diff['diff_gt_rank_count']} / {diff['total_queries']} ({diff['diff_gt_rank_count']/diff['total_queries']*100:.2f}%)")
    print(f"    + Số truy vấn BM25 xếp đúng hơn TF-IDF: {diff['b_better_than_a']}")
    print(f"    + Số truy vấn TF-IDF xếp đúng hơn BM25: {diff['a_better_than_b']}")

    print("\nThời gian truy vấn trung bình (ms / truy vấn):")
    print(f"  + TF-IDF (Trọn gói gồm tiền xử lý) : {tfidf_ms_full:.4f} ms | Chỉ truy vấn & xếp hạng: {tfidf_ms_rank:.4f} ms")
    print(f"  + BM25   (Trọn gói gồm tiền xử lý) : {bm25_ms_full:.4f} ms | Chỉ truy vấn & xếp hạng: {bm25_ms_rank:.4f} ms")

    if diff["diff_gt_details"]:
        print("\nDanh sách chi tiết các truy vấn có thứ hạng tài liệu đích khác nhau:")
        print(f"{'QID':>4} | {'GT':>3} | {'Rank TF-IDF':>11} | {'Rank BM25':>9} | {'Thắng':<7} | {'Nội dung câu truy vấn'}")
        print("-" * 95)
        for item in diff["diff_gt_details"]:
            q_short = item["query_text"][:52] + ("..." if len(item["query_text"]) > 52 else "")
            print(
                f"{item['query_id']:>4} | D{item['ground_truth_doc_id']:<2} | "
                f"{item['rank_a']:>11} | {item['rank_b']:>9} | {item['winner']:<7} | {q_short}"
            )

    return {
        "index_field": index_field,
        "bm25_params": {"k1": best_k1, "b": best_b},
        "diff_summary": {
            "total_queries": diff["total_queries"],
            "diff_top1_count": diff["diff_top1_count"],
            "diff_top5_count": diff["diff_topk_count"],
            "diff_gt_rank_count": diff["diff_gt_rank_count"],
            "bm25_better_count": diff["b_better_than_a"],
            "tfidf_better_count": diff["a_better_than_b"],
        },
        "timing_ms": {
            "tfidf_full_ms": round(tfidf_ms_full, 4),
            "bm25_full_ms": round(bm25_ms_full, 4),
            "tfidf_rank_only_ms": round(tfidf_ms_rank, 4),
            "bm25_rank_only_ms": round(bm25_ms_rank, 4),
        },
        "diff_gt_details": diff["diff_gt_details"]
    }


def main() -> None:
    corpus, queries = load_dataset(DATA_FILE)
    print(f"Đã nạp bộ dữ liệu Tuần 4: {len(corpus)} tài liệu (60 gốc + 30 gây nhiễu) & {len(queries)} câu truy vấn.\n")

    exp1 = run_index_field_experiment(corpus, queries, repeats=30)
    exp2 = run_bm25_grid_search(corpus, queries, index_field="both")

    # So sánh TF-IDF với cả BM25 mặc định (1.5, 0.75) và BM25 tối ưu từ Grid Search
    exp3_default = run_ranking_diff_and_timing(corpus, queries, index_field="both", best_k1=1.5, best_b=0.75, repeats=50)

    best_k1 = exp2["best"]["k1"]
    best_b = exp2["best"]["b"]
    exp3_best = None
    if (best_k1, best_b) != (1.5, 0.75):
        exp3_best = run_ranking_diff_and_timing(corpus, queries, index_field="both", best_k1=best_k1, best_b=best_b, repeats=50)

    summary_out = {
        "dataset": DATA_FILE,
        "total_docs": len(corpus),
        "total_queries": len(queries),
        "experiment_1_index_field": exp1,
        "experiment_2_bm25_grid": exp2,
        "experiment_3_diff_and_timing_default": exp3_default,
        "experiment_3_diff_and_timing_best": exp3_best,
    }
    with open("ket_qua_tuan4.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print("\nĐã lưu toàn bộ kết quả thực nghiệm ra file 'ket_qua_tuan4.json'.")


if __name__ == "__main__":
    main()
