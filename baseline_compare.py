# -*- coding: utf-8 -*-
"""
Module 5: Đối chứng kết quả với thư viện chuẩn (scikit-learn & rank_bm25)
Đồ án Cơ sở ngành CNTT - Đề tài : Xây dựng chỉ mục ngược và xếp hạng TF-IDF / BM25
Sinh viên thực hiện: Phan Nguyễn Ngọc Thiện - 66133552
"""

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocessor import VietnamesePreprocessor
from inverted_index import InvertedIndex
from ranker import TFIDFRanker


def verify_sklearn_exact_match() -> None:
    """
    Kiểm chứng cấu hình TfidfVectorizer(sublinear_tf=True, smooth_idf=True, norm=None)
    khớp tuyệt đối đến 4 chữ số thập phân với TFIDFRanker tự cài trên ví dụ Chương 3.
    """
    preprocessor = VietnamesePreprocessor(use_external_tokenizer=False)
    mini_corpus = [
        {"doc_id": 1, "text": "Mức học phí ngành công nghệ thông tin hệ đại trà."},
        {"doc_id": 2, "text": "Điểm chuẩn trúng tuyển ngành công nghệ thông tin."},
        {"doc_id": 3, "text": "Quy định đóng học phí, miễn giảm học phí và chính sách học phí cho sinh viên diện chính sách."}
    ]
    query = "Học phí công nghệ thông tin"

    # 1. Điểm từ TFIDFRanker tự cài
    idx = InvertedIndex(preprocessor=preprocessor)
    idx.build_index(mini_corpus)
    custom_tfidf = TFIDFRanker(idx)
    custom_scores = custom_tfidf.score_all(query)

    # 2. Điểm từ scikit-learn TfidfVectorizer cấu hình tương đương
    preprocessed_docs: List[str] = [" ".join(idx.doc_tokens[d["doc_id"]]) for d in mini_corpus]
    preprocessed_query: str = " ".join(preprocessor.preprocess(query))

    vec = TfidfVectorizer(
        sublinear_tf=True,
        smooth_idf=True,
        use_idf=True,
        norm=None,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False
    )
    X_tfidf = vec.fit_transform(preprocessed_docs)

    q_cnt = CountVectorizer(
        vocabulary=vec.vocabulary_,
        binary=True,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False
    )
    q_vec = q_cnt.transform([preprocessed_query])
    sklearn_scores = X_tfidf.dot(q_vec.T).toarray().ravel()

    print("=" * 75)
    print("ĐỐI CHỨNG ĐIỂM SỐ: TFIDFRanker (Tự cài) vs. scikit-learn TfidfVectorizer")
    print("=" * 75)
    for i, doc in enumerate(mini_corpus):
        d_id = doc["doc_id"]
        c_score = custom_scores.get(d_id, 0.0)
        s_score = float(sklearn_scores[i])
        diff = abs(c_score - s_score)
        print(f"Doc D{d_id}: Tự cài = {c_score:.6f} | scikit-learn = {s_score:.6f} | Sai số = {diff:.2e}")
    print("=" * 75)


if __name__ == "__main__":
    verify_sklearn_exact_match()
