from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

class RoughService:
    def calculate_rough(self, references, candidates):
        return scorer.score(references, candidates)
