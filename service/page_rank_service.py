import numpy as np


class PageRankService:
    def cal_page_rank_score(self, similarity_matrix, max_iter=200, d=0.85, beta=0.15, tol=1.0e-8):
        num_sentences = similarity_matrix.shape[0]
        pagerank = np.ones(num_sentences) / num_sentences
        for _ in range(max_iter):
            new_pagerank = (1 - d) / num_sentences + d * similarity_matrix.T @ (
                    pagerank / similarity_matrix.sum(axis=1))
            if np.linalg.norm(new_pagerank - pagerank, ord=1) < tol:
                break
            pagerank = new_pagerank
        return pagerank
