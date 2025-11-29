import json
import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from normalize import normalize
from model_loader import ModelLoader
import spacy
from spacy.lang.en import English


class SreRAG:
    """Performs Sentence Retrieval and Evidence ranking."""
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize SreRAG with pre-loaded models.
        
        Args:
            model_loader: An instance of ModelLoader with loaded models
        """
        self.model_loader = model_loader
        self.device = model_loader.device
        self.searcher = model_loader.searcher
        self.reader = model_loader.reader
        self.cls_tokenizer = model_loader.cls_tokenizer
        self.cls_model = model_loader.cls_model
        self.ret_tokenizer = model_loader.ret_tokenizer
        self.dense_model = model_loader.dense_model
        self.nlp = spacy.load("en_core_web_sm")

    
    def spacy_preprocess_query(self, question: str) -> str:
        """
        Preprocess question to improve BM25 retrieval.
        - keep important content words (nouns, verbs, adjectives, proper nouns)
        - remove stopwords / punctuation
        - lemmatize
        - boost named entities (e.g., person names) by repeating them
        """
        doc = self.nlp(question)
        tokens = []

        # keep important content words
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if token.is_stop:
                continue
            if token.pos_ not in {"NOUN", "PROPN", "VERB", "ADJ"}:
                continue

            lemma = (token.lemma_ or token.text).lower().strip()
            if lemma:
                tokens.append(lemma)

        # Boost named entities (subjects are critical for PopQA)
        for ent in doc.ents:
            ent_text = ent.text.lower().strip()
            if ent_text:
                tokens.append(ent_text)
                tokens.append(ent_text)  # duplicate to give more weight

        processed = " ".join(tokens)

        # Safety: if preprocessing killed everything, fall back to original question
        return processed if processed.strip() else question


    def bm25_retrieve(self, query_text, k=10): 
        """Raw BM25 retrieval with SpaCy-preprocessed query."""
        processed_query = self.spacy_preprocess_query(query_text)
        print(f"Query: {query_text}  |  Processed: {processed_query}")
        hits = self.searcher.search(processed_query, k)
        results = []

        for rank, h in enumerate(hits, start=1):
            docid = h.docid
            raw_json = self.searcher.doc(docid).raw()
            record = json.loads(raw_json)
            text_sample = record.get("contents", "")
            results.append({
                "docid": int(docid),
                "score": float(h.score),
                "rank": rank,
                "text": text_sample
            })
        return results
    
    def bm25_retrieve_and_rank(self, question: str, bm25_k: int = 100):
        """
        Retrieve and rank using only BM25.
        
        Args:
            question: The query question
            bm25_k: Number of candidates to retrieve with BM25
        
        Returns:
            list: Array of BM25 results with normalized text
        """
        bm25_results = self.bm25_retrieve(question, k=bm25_k)
        
        processed_results = []
        for result in bm25_results:
            text = result["text"]
            # Remove the title part
            if text.startswith('"') and '"\n' in text:
                text = text.split('"\n', 1)[1]
            # Normalize the text
            text = normalize(text)
            
            processed_results.append({
                "rank": result["rank"],
                "docid": result["docid"],
                "bm25_score": result["score"],
                "sentence": text
            })
        
        return processed_results
    
    def dense_retrieve_and_rank(self, question: str, bm25_results: list, ):
        """
        Rank BM25 results using dense retrieval (Contriever).
        
        Args:
            question: The query question
            bm25_results: List of dictionaries from bm25_retrieve_and_rank
            truncate_len: Length to truncate sentences in display
        
        Returns:
            list: Array of dictionaries containing dense ranking information
        """
        if not bm25_results:
            return []
        
        candidate_sentences = [result["sentence"] for result in bm25_results]
        
        # Encode query and candidates with Contriever
        query_emb = self.encode_texts([question])
        cand_embs = self.encode_texts(candidate_sentences)
        
        scores = (query_emb @ cand_embs.T)[0]
        sorted_indices = torch.argsort(scores, descending=True).tolist()
        
        results = []
        for rank, idx in enumerate(sorted_indices, start=1):
            sent = candidate_sentences[idx]
            original_bm25 = bm25_results[idx]
            
            results.append({
                "dense_rank": rank,
                "bm25_rank": original_bm25["rank"],
                "docid": original_bm25["docid"],
                "bm25_score": original_bm25["bm25_score"],
                "dense_score": scores[idx].item(),
                "sentence": sent
            })
        
        return results
    
    def fuse_bm25_and_dense(
        self,
        dense_results: list,
        k_rrf: int = 60,
        top_k: int | None = None,
    ) -> list:
        """
        Fuse BM25 and dense ranks using Reciprocal Rank Fusion (RRF).

        Args:
            dense_results: output of dense_retrieve_and_rank
                           (must contain 'bm25_rank' and 'dense_rank')
            k_rrf: RRF constant (e.g., 60)
            top_k: if not None, keep only top_k documents after fusion

        Returns:
            list of result dicts, sorted by fused score (rrf_score desc)
        """
        if not dense_results:
            return []

        fused = []
        for r in dense_results:
            bm25_rank = r["bm25_rank"]
            dense_rank = r["dense_rank"]

            rrf_score = 1.0 / (k_rrf + bm25_rank) + 1.0 / (k_rrf + dense_rank)

            new_r = dict(r)          # copy to avoid mutating original
            new_r["rrf_score"] = rrf_score
            fused.append(new_r)

        # Sort by fused RRF score (higher is better)
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Optionally cut to top_k documents
        if top_k is not None:
            fused = fused[:top_k]

        return fused


    def score_candidates( self,  question: str, dense_results: list ) -> list:
        if not dense_results:
            return []

        # Build full context from all candidate docs (not split)
        doc_texts = [r["sentence"] for r in dense_results]
        full_context = " ".join(doc_texts)

        scored_results = []
        sentence_id = 0

        # IMPORTANT: iterate over dense_results (dicts), not over raw strings
        for r in dense_results:
            doc = self.nlp(r["sentence"])
            for sent in doc.sents:
                small_sent = sent.text.strip()
                if not small_sent:
                    continue

                sentence_id += 1
                p_no, p_yes = self.score_sentence( question, full_context, small_sent )

                scored_results.append({
                    "sentence_id": sentence_id,
                    "dense_rank": r["dense_rank"],
                    "bm25_rank": r["bm25_rank"],
                    "docid": r["docid"],
                    "bm25_score": r["bm25_score"],
                    "dense_score": r["dense_score"],
                    "p_yes": p_yes,
                    "p_no": p_no, 
                    "sentence": small_sent,
                    **({"rrf_score": r["rrf_score"]} if "rrf_score" in r else {}),
                })

        return scored_results


    def build_text(self, question: str, full_context: str, sentence: str) -> str:
        """Build input text for classifier."""
        return (
            f"Query:\n{question}\n\n"
            f"Full context:\n{full_context}\n\n"
            f"Candidate sentence:\n{sentence}\n\n"
            "Is this sentence useful in answering the query? Answer Yes or No."
        )

    @torch.no_grad()
    def score_sentence(self, question: str, full_context: str, sentence: str,
                       threshold: float = 0.5, max_length: int = 1024):
        """Score a single sentence using the classifier model."""
        text = self.build_text(question, full_context, sentence)
        enc = self.cls_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.cls_model(**enc)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]

        p_no = probs[0].item()
        p_yes = probs[1].item() 
        return p_no, p_yes

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling for dense embeddings."""
        mask = attention_mask.unsqueeze(-1).bool()
        token_embeddings = token_embeddings.masked_fill(~mask, 0.0)
        summed = token_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    @torch.no_grad()
    def encode_texts(self, texts, max_length: int = 512):
        """Encode texts using Contriever."""
        enc = self.ret_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.dense_model(**enc)
        embeddings = self.mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
 
 
    
    

    def print_ranking_table(self, results: list, show_scoring: bool = False):
        """
        Print results in a PrettyTable format.
        
        Args:
            results: List of result dictionaries
            show_scoring: Whether to show P(Yes), P(No), Pred columns
        """
        if show_scoring and results and "p_yes" in results[0]:
            table = PrettyTable()
            table.field_names = [
                # "Dense Rank", "BM25 Rank", "DocID", 
                # "BM25 Score", "Dense Score",

                "P(Yes)", "P(No)", "Pred",
                "Sentence",
            ]
            
            for res in results:
                table.add_row([
                    # res["dense_rank"],
                    # res["bm25_rank"],
                    # res["docid"],
                    # f"{res['bm25_score']:.4f}",
                    # f"{res['dense_score']:.4f}",
                    f"{res['p_yes']:.4f}",
                    f"{res['p_no']:.4f}",
                    res["pred"],
                    res["sentence"],
                ])
            
            # Align all columns to the left
            table.align = "l"
            
        else:
            table = PrettyTable()
            table.field_names = [
                "Dense Rank", 
                "BM25 Rank", 
                "DocID", 
                "BM25 Score", 
                "Dense Score",
                "Sentence",
            ]
            
            for res in results:
                table.add_row([
                    res["dense_rank"],
                    res["bm25_rank"],
                    res["docid"],
                    f"{res['bm25_score']:.4f}",
                    f"{res['dense_score']:.4f}",
                    res["sentence"],
                ])
            
            # Align all columns to the left
            table.align = "l"
        
        table.max_width = 500
        print(table)