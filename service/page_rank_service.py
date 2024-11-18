import networkx as nx
import numpy as np


class PageRankService:
    def cal_page_rank_score(self, similarity_matrix, max_iter=100, d=0.85, beta=0.15, tol=1.0e-8):
        graph = nx.from_numpy_array(similarity_matrix)
        pageranks = nx.pagerank(graph)
        # N = len(graph)
        #
        # print(f'N: {N}')
        # pageranks = np.full(N, 1.0 / N)
        # new_pageranks = np.zeros(N)
        #
        # for iteration in range(max_iter):
        #     for u in range(N):
        #         # Calculate the sum part of the formula
        #         rank_sum = 0
        #         for v in range(N):
        #             if similarity_matrix[v][u] > 0:  # if there's a link from v to u
        #                 rank_sum += pageranks[v] / np.sum(similarity_matrix[v])  # divide by out-degree of v
        #         # Apply the PageRank formula
        #         new_pageranks[u] = (1 - d) / N + d * rank_sum
        #
        #     # Check for convergence
        #     if np.linalg.norm(new_pageranks - pageranks, ord=1) < tol:
        #         print(f"Converged after {iteration} iterations.")
        #         break
        #
        #     pageranks = new_pageranks.copy()

        return pageranks
