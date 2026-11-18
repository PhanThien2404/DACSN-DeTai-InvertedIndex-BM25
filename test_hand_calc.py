# -*- coding: utf-8 -*-
"""
Unit Test: Kiểm thử tự động tái hiện chính xác 100% Ví dụ tính tay ở Chương 3
và chạy kiểm thử hệ thống trên bộ dữ liệu mở rộng Tuần 3 (60 tài liệu & 120 truy vấn).
Đồ án Cơ sở ngành CNTT - Đề tài : Xây dựng chỉ mục ngược và xếp hạng TF-IDF / BM25
Sinh viên thực hiện: Phan Nguyễn Ngọc Thiện - 66133552
"""

import unittest
import json
import os
from preprocessor import VietnamesePreprocessor
from inverted_index import InvertedIndex
from ranker import TFIDFRanker, BM25Ranker
from evaluator import Evaluator


class TestChapter3HandCalculation(unittest.TestCase):
    """
    Bộ Unit Test kiểm chứng từng bước ví dụ tính tay ở Chương 3 trong Báo cáo:
      - D1: "Mức học phí ngành công nghệ thông tin hệ đại trà." -> 5 từ
      - D2: "Điểm chuẩn trúng tuyển ngành công nghệ thông tin." -> 4 từ
      - D3: "Quy định đóng học phí, miễn giảm học phí và chính sách học phí cho sinh viên diện chính sách." -> 9 từ
      - Query Q: "Học phí công nghệ thông tin" -> ["học_phí", "công_nghệ_thông_tin"]
      - avgdl = (5 + 4 + 9) / 3 = 6.0
      - Điểm TF-IDF: D3 (2.7024) > D1 (2.5754) > D2 (1.2877) -> D3 chiếm Hạng 1 sai lệch
      - Điểm BM25 (k1=1.5, b=0.75): D1 (1.0162) > D3 (0.6963) > D2 (0.5529) -> Đảo ngôi trả D1 về Hạng 1!
    """

    def setUp(self) -> None:
        # Khởi tạo Preprocessor tắt pyvi ngoài để đảm bảo tách từ khớp tuyệt đối với 12 từ vựng ở Bảng 3.3
        self.preprocessor = VietnamesePreprocessor(use_external_tokenizer=False)
        self.mini_corpus = [
            {
                "doc_id": 1,
                "text": "Mức học phí ngành công nghệ thông tin hệ đại trà."
            },
            {
                "doc_id": 2,
                "text": "Điểm chuẩn trúng tuyển ngành công nghệ thông tin."
            },
            {
                "doc_id": 3,
                "text": "Quy định đóng học phí, miễn giảm học phí và chính sách học phí cho sinh viên diện chính sách."
            }
        ]
        self.query = "Học phí công nghệ thông tin"

        self.index = InvertedIndex(preprocessor=self.preprocessor)
        self.index.build_index(self.mini_corpus)

        self.tfidf_ranker = TFIDFRanker(self.index)
        self.bm25_ranker = BM25Ranker(self.index, k1=1.5, b=0.75)

    def test_step1_preprocessing_and_avgdl(self) -> None:
        """Kiểm tra Bước 1 (Mục 3.2): Tiền xử lý tách từ, độ dài |D1|=5, |D2|=4, |D3|=9 và avgdl=6.0."""
        self.assertEqual(
            self.index.doc_tokens[1],
            ["mức", "học_phí", "ngành", "công_nghệ_thông_tin", "đại_trà"]
        )
        self.assertEqual(
            self.index.doc_tokens[2],
            ["điểm_chuẩn", "trúng_tuyển", "ngành", "công_nghệ_thông_tin"]
        )
        self.assertEqual(
            self.index.doc_tokens[3],
            ["quy_định", "đóng", "học_phí", "miễn_giảm", "học_phí", "chính_sách", "học_phí", "sinh_viên", "chính_sách"]
        )
        self.assertEqual(self.index.doc_lengths[1], 5)
        self.assertEqual(self.index.doc_lengths[2], 4)
        self.assertEqual(self.index.doc_lengths[3], 9)
        self.assertAlmostEqual(self.index.avgdl, 6.0, places=4)

    def test_step2_inverted_index_postings(self) -> None:
        """Kiểm tra Bước 2 (Mục 3.3): Bảng Chỉ mục ngược gồm 12 từ vựng và Posting List."""
        self.assertEqual(len(self.index.vocab), 12)
        self.assertEqual(self.index.get_df("công_nghệ_thông_tin"), 2)
        self.assertEqual(self.index.get_postings("công_nghệ_thông_tin"), [(1, 1), (2, 1)])

        self.assertEqual(self.index.get_df("học_phí"), 2)
        self.assertEqual(self.index.get_postings("học_phí"), [(1, 1), (3, 3)])

        self.assertEqual(self.index.get_df("chính_sách"), 1)
        self.assertEqual(self.index.get_postings("chính_sách"), [(3, 2)])

    def test_step3_tfidf_scores_and_ranking(self) -> None:
        """Kiểm tra Bước 3 (Mục 3.4): Điểm số và thứ hạng của thuật toán TF-IDF."""
        idf_hp = self.tfidf_ranker.compute_idf("học_phí")
        idf_cntt = self.tfidf_ranker.compute_idf("công_nghệ_thông_tin")
        self.assertAlmostEqual(idf_hp, 1.2877, places=4)
        self.assertAlmostEqual(idf_cntt, 1.2877, places=4)

        scores = self.tfidf_ranker.score_all(self.query)
        self.assertAlmostEqual(scores[1], 2.5754, places=3)
        self.assertAlmostEqual(scores[2], 1.2877, places=3)
        self.assertAlmostEqual(scores[3], 2.7024, places=3)

        # Kiểm tra thứ hạng TF-IDF: D3 (Hạng 1) > D1 (Hạng 2) > D2 (Hạng 3)
        ranked = self.tfidf_ranker.rank(self.query, top_k=3)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        self.assertEqual(ranked_doc_ids, [3, 1, 2])

    def test_step4_bm25_scores_and_rank_reversal(self) -> None:
        """Kiểm tra Bước 4 (Mục 3.5): Điểm số Okapi BM25 và sự đảo ngược thứ hạng D1 > D3 > D2."""
        idf_bm25 = self.bm25_ranker.compute_idf("học_phí")
        self.assertAlmostEqual(idf_bm25, 0.4700, places=4)

        # Kiểm tra hệ số chuẩn hóa độ dài K_D
        self.assertAlmostEqual(self.bm25_ranker.compute_kd(1), 1.3125, places=4)
        self.assertAlmostEqual(self.bm25_ranker.compute_kd(2), 1.1250, places=4)
        self.assertAlmostEqual(self.bm25_ranker.compute_kd(3), 2.0625, places=4)

        # Kiểm tra tổng điểm BM25
        scores = self.bm25_ranker.score_all(self.query)
        self.assertAlmostEqual(scores[1], 1.0162, places=3)
        self.assertAlmostEqual(scores[2], 0.5529, places=3)
        self.assertAlmostEqual(scores[3], 0.6963, places=3)

        # Kiểm tra BM25 đảo ngược thứ hạng so với TF-IDF: D1 (Hạng 1) > D3 (Hạng 2) > D2 (Hạng 3)
        ranked = self.bm25_ranker.rank(self.query, top_k=3)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        self.assertEqual(ranked_doc_ids, [1, 3, 2])

    def test_step5_evaluation_metrics_recall_and_mrr(self) -> None:
        """Kiểm tra Bước 5 (Mục 3.6): Độ đo Recall@1, Recall@3 và MRR giữa TF-IDF và BM25."""
        # Với Q1 (rank_TFIDF = 2, rank_BM25 = 1) và Q2 giả định (rank = 3)
        ranks_tfidf = [2, 3]
        ranks_bm25 = [1, 3]

        self.assertAlmostEqual(Evaluator.recall_at_k(ranks_tfidf, 1), 0.0, places=4)
        self.assertAlmostEqual(Evaluator.recall_at_k(ranks_bm25, 1), 0.5, places=4)

        self.assertAlmostEqual(Evaluator.recall_at_k(ranks_tfidf, 3), 1.0, places=4)
        self.assertAlmostEqual(Evaluator.recall_at_k(ranks_bm25, 3), 1.0, places=4)

        self.assertAlmostEqual(Evaluator.mrr(ranks_tfidf), 5.0 / 12.0, places=4)  # ~0.4167
        self.assertAlmostEqual(Evaluator.mrr(ranks_bm25), 2.0 / 3.0, places=4)    # ~0.6667


def run_dry_run_display() -> None:
    """In bảng kết quả đối chiếu trực quan giữa tính tay Chương 3 và kết quả chạy Code."""
    print("=" * 80)
    print("1. KIỂM CHỨNG VÍ DỤ TÍNH TAY CHƯƠNG 3 (DRY-RUN VERIFICATION)")
    print("=" * 80)
    preprocessor = VietnamesePreprocessor(use_external_tokenizer=False)
    mini_corpus = [
        {"doc_id": 1, "text": "Mức học phí ngành công nghệ thông tin hệ đại trà."},
        {"doc_id": 2, "text": "Điểm chuẩn trúng tuyển ngành công nghệ thông tin."},
        {"doc_id": 3, "text": "Quy định đóng học phí, miễn giảm học phí và chính sách học phí cho sinh viên diện chính sách."}
    ]
    query = "Học phí công nghệ thông tin"

    idx = InvertedIndex(preprocessor=preprocessor)
    idx.build_index(mini_corpus)

    tfidf = TFIDFRanker(idx)
    bm25 = BM25Ranker(idx, k1=1.5, b=0.75)

    print(f"- Tổng số tài liệu N = {idx.total_docs}, Độ dài trung bình avgdl = {idx.avgdl:.4f}")
    for d_id in [1, 2, 3]:
        print(f"  + D{d_id} (|D{d_id}|={idx.doc_lengths[d_id]}): {idx.doc_tokens[d_id]}")

    tfidf_scores = tfidf.score_all(query)
    bm25_scores = bm25.score_all(query)
    tfidf_ranks = [d for d, _ in tfidf.rank(query, top_k=3)]
    bm25_ranks = [d for d, _ in bm25.rank(query, top_k=3)]

    print("\nBảng đối chiếu điểm số từ Code Python vs. Tính tay Chương 3:")
    print(f"{'Tài liệu':<12} | {'TF-IDF (Code)':<15} | {'Hạng TF-IDF':<13} | {'BM25 (Code)':<13} | {'Hạng BM25':<10}")
    print("-" * 75)
    for d_id in [1, 3, 2]:
        r_tfidf = tfidf_ranks.index(d_id) + 1
        r_bm25 = bm25_ranks.index(d_id) + 1
        print(f"Doc D{d_id:<7} | {tfidf_scores[d_id]:<15.4f} | Hạng {r_tfidf:<8} | {bm25_scores[d_id]:<13.4f} | Hạng {r_bm25:<5}")
    print("-" * 75)

    # Nếu có file faq_data_Tuan3.json thì chạy luôn đánh giá thực nghiệm bước đầu
    if os.path.exists("faq_data_Tuan3.json"):
        print("\n" + "=" * 80)
        print("2. CHẠY THỬ NGHIỆM TRÊN BỘ DỮ LIỆU MỞ RỘNG TUẦN 3 (60 DOCS & 120 QUERIES)")
        print("=" * 80)
        with open("faq_data_Tuan3.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        corpus_60 = data["corpus"]
        queries_120 = data["test_queries"]

        full_idx = InvertedIndex(preprocessor=VietnamesePreprocessor(), index_field="both")
        full_idx.build_index(corpus_60)
        print(f"- Đã lập chỉ mục {full_idx.total_docs} tài liệu FAQ | Kích thước từ điển V = {len(full_idx.vocab)} từ | avgdl = {full_idx.avgdl:.2f} từ.")

        tfidf_full = TFIDFRanker(full_idx)
        bm25_full = BM25Ranker(full_idx, k1=1.5, b=0.75)

        m_tfidf = Evaluator.evaluate_ranker(tfidf_full, queries_120, k_values=(1, 3, 5, 10))
        m_bm25 = Evaluator.evaluate_ranker(bm25_full, queries_120, k_values=(1, 3, 5, 10))

        print(f"\n{'Thuật toán':<15} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<10}")
        print("-" * 78)
        print(f"{'TF-IDF (Tự cài)':<15} | {m_tfidf['Recall@1']*100:>8.2f}% | {m_tfidf['Recall@3']*100:>8.2f}% | {m_tfidf['Recall@5']*100:>8.2f}% | {m_tfidf['Recall@10']*100:>8.2f}% | {m_tfidf['MRR']:>8.4f}")
        print(f"{'BM25 (Tự cài)':<15} | {m_bm25['Recall@1']*100:>8.2f}% | {m_bm25['Recall@3']*100:>8.2f}% | {m_bm25['Recall@5']*100:>8.2f}% | {m_bm25['Recall@10']*100:>8.2f}% | {m_bm25['MRR']:>8.4f}")
        print("-" * 78)


if __name__ == "__main__":
    run_dry_run_display()
    print("\n" + "=" * 80)
    print("3. CHẠY UNIT TEST TỰ ĐỘNG (UNITTEST FRAMEWORK)")
    print("=" * 80)
    unittest.main(verbosity=2)
