"""
ranking.py — Tech0 Search v1.0
TF-IDF ベースの検索エンジン（SearchEngine クラス）を提供する。
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from datetime import datetime

class SearchEngine:
    """TF-IDFベースの検索エンジン（エラー耐性強化版）"""

    def __init__(self):
        # どんなデータ件数でもエラーにならない最強の設定
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,            # 1件のドキュメントでも計算を許可
            max_df=1.0,          # 全出現率を許可
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b" # 1文字単語も許可
        )
        self.tfidf_matrix = None
        self.pages = []
        self.is_fitted = False

    def build_index(self, pages: list):
        if not pages:
            return
        self.pages = pages
        corpus = []
        for p in pages:
            kw = p.get("keywords", "") or ""
            kw_list = kw.split(",") if isinstance(kw, str) else kw
            text = " ".join([
                (p.get("title", "") + " ") * 3,
                (p.get("description", "") + " ") * 2,
                (p.get("full_text", "") + " "),
                (" ".join(kw_list) + " ") * 2,
            ])
            corpus.append(text)

        # データが空の場合は処理をスキップ
        if corpus:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            self.is_fitted = True

    def search(self, query: str, top_n: int = 20) -> list:
        if not self.is_fitted or not query.strip():
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        results = []
        for idx, base_score in enumerate(similarities):
            if base_score > 0.001: # 閾値を下げてヒットしやすくする
                page = self.pages[idx].copy()
                final_score = self._calculate_final_score(page, base_score, query)
                page["relevance_score"] = round(float(final_score) * 100, 1)
                results.append(page)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_n]

    def _calculate_final_score(self, page: dict, base_score: float, query: str) -> float:
        score = base_score
        query_lower = query.lower()
        title = page.get("title", "").lower()
        if query_lower in title:
            score *= 1.5
        return score

# ── シングルトン管理 ──────────────────────────────────────────
_engine = None

def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine

def rebuild_index(pages: List[dict]):
    engine = get_engine()
    engine.build_index(pages)