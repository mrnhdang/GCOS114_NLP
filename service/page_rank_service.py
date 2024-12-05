import numpy as np


class PageRankService:
    def cal_page_rank_score(self, similarity_matrix, clean_data_arr, max_iter=200, d=0.85, beta=0.15, tol=1.0e-8):
        pagerank = {}
        for data_index, documents in enumerate(clean_data_arr):
            num_sentences = similarity_matrix[data_index].shape[0]
            pagerank.update({data_index: np.ones(num_sentences) / num_sentences})
            for _ in range(max_iter):
                new_pagerank = (1 - d) / num_sentences + d * similarity_matrix[data_index].T @ (
                            pagerank[data_index] / similarity_matrix[data_index].sum(axis=1))
                if np.linalg.norm(new_pagerank - pagerank[data_index], ord=1) < tol:
                    break
                pagerank[data_index] = new_pagerank
        return pagerank
