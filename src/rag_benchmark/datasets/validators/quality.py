"""Quality validation utilities for datasets"""

import math
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import re

from rag_benchmark.datasets.schemas.golden import GoldenRecord, CorpusRecord
from rag_benchmark.datasets.loaders.base import ValidationResult


class QualityValidator:
    """Validates dataset quality metrics"""
    
    @staticmethod
    def analyze_question_quality(questions: List[str]) -> Dict[str, Any]:
        """Analyze question quality
        
        Args:
            questions: List of question strings
            
        Returns:
            Dictionary with quality metrics
        """
        if not questions:
            return {}
        
        lengths = [len(q) for q in questions]
        word_counts = [len(q.split()) for q in questions]
        
        # Question type patterns
        wh_patterns = [
            r'^\b(what|when|where|who|why|how)\b',
            r'\?$'
        ]
        
        wh_question_count = 0
        for q in questions:
            q_lower = q.lower()
            if any(re.search(pattern, q_lower) for pattern in wh_patterns):
                wh_question_count += 1
        
        return {
            "count": len(questions),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "min_word_count": min(word_counts),
            "max_word_count": max(word_counts),
            "wh_question_ratio": wh_question_count / len(questions),
            "short_questions": sum(1 for l in lengths if l < 50),
            "long_questions": sum(1 for l in lengths if l > 500)
        }
    
    @staticmethod
    def analyze_answer_quality(answers: List[str]) -> Dict[str, Any]:
        """Analyze answer quality
        
        Args:
            answers: List of answer strings
            
        Returns:
            Dictionary with quality metrics
        """
        if not answers:
            return {}
        
        lengths = [len(a) for a in answers]
        word_counts = [len(a.split()) for a in answers]
        
        # Answer patterns
        has_complete_sentence = sum(1 for a in answers if re.search(r'[.!?]\s*$', a))
        has_numbers = sum(1 for a in answers if re.search(r'\d', a))
        has_quotes = sum(1 for a in answers if re.search(r'"[^"]*"', a))
        
        return {
            "count": len(answers),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "complete_sentence_ratio": has_complete_sentence / len(answers),
            "contains_numbers_ratio": has_numbers / len(answers),
            "contains_quotes_ratio": has_quotes / len(answers),
            "short_answers": sum(1 for l in lengths if l < 10),
            "long_answers": sum(1 for l in lengths if l > 1000)
        }
    
    @staticmethod
    def analyze_context_quality(contexts_list: List[List[str]]) -> Dict[str, Any]:
        """Analyze context quality
        
        Args:
            contexts_list: List of context lists (one list per golden record)
            
        Returns:
            Dictionary with quality metrics
        """
        if not contexts_list:
            return {}
        
        all_contexts = [ctx for contexts in contexts_list for ctx in contexts]
        context_counts = [len(contexts) for contexts in contexts_list]
        context_lengths = [len(ctx) for ctx in all_contexts]
        
        # Context coverage metrics
        avg_coverage = sum(context_lengths) / len(contexts_list) if contexts_list else 0
        
        return {
            "total_contexts": len(all_contexts),
            "avg_contexts_per_record": sum(context_counts) / len(context_counts),
            "min_contexts_per_record": min(context_counts),
            "max_contexts_per_record": max(context_counts),
            "avg_context_length": sum(context_lengths) / len(context_lengths) if context_lengths else 0,
            "min_context_length": min(context_lengths) if context_lengths else 0,
            "max_context_length": max(context_lengths) if context_lengths else 0,
            "avg_coverage_per_record": avg_coverage,
            "empty_contexts": sum(1 for ctx in all_contexts if not ctx.strip()),
            "short_contexts": sum(1 for l in context_lengths if l < 50),
            "long_contexts": sum(1 for l in context_lengths if l > 5000)
        }
    
    @staticmethod
    def compute_relevance_scores(
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]]
    ) -> Dict[str, Any]:
        """Compute simple relevance metrics between Q&A and contexts
        
        Args:
            questions: List of questions
            answers: List of answers
            contexts_list: List of context lists
            
        Returns:
            Dictionary with relevance metrics
        """
        if not questions or not answers:
            return {}
        
        # Simple keyword overlap metrics
        qa_overlap_scores = []
        ctx_overlap_scores = []
        
        for i, (q, a) in enumerate(zip(questions, answers)):
            # Q-A overlap
            q_words = set(q.lower().split())
            a_words = set(a.lower().split())
            overlap = len(q_words & a_words)
            union = len(q_words | a_words)
            qa_overlap = overlap / union if union > 0 else 0
            qa_overlap_scores.append(qa_overlap)
            
            # Q-Context overlap
            if i < len(contexts_list) and contexts_list[i]:
                ctx_text = " ".join(contexts_list[i]).lower()
                ctx_words = set(ctx_text.split())
                overlap = len(q_words & ctx_words)
                union = len(q_words | ctx_words)
                ctx_overlap = overlap / union if union > 0 else 0
                ctx_overlap_scores.append(ctx_overlap)
        
        return {
            "avg_qa_overlap": sum(qa_overlap_scores) / len(qa_overlap_scores) if qa_overlap_scores else 0,
            "min_qa_overlap": min(qa_overlap_scores) if qa_overlap_scores else 0,
            "max_qa_overlap": max(qa_overlap_scores) if qa_overlap_scores else 0,
            "avg_qc_overlap": sum(ctx_overlap_scores) / len(ctx_overlap_scores) if ctx_overlap_scores else 0,
            "min_qc_overlap": min(ctx_overlap_scores) if ctx_overlap_scores else 0,
            "max_qc_overlap": max(ctx_overlap_scores) if ctx_overlap_scores else 0
        }
    
    @staticmethod
    def validate_dataset_quality(
        golden_records: List[GoldenRecord],
        corpus_records: List[CorpusRecord]
    ) -> ValidationResult:
        """Validate overall dataset quality
        
        Args:
            golden_records: List of GoldenRecord objects
            corpus_records: List of CorpusRecord objects
            
        Returns:
            ValidationResult with quality metrics and warnings
        """
        result = ValidationResult()
        
        if not golden_records:
            result.add_error("No golden records to analyze")
            return result
        
        # Extract data for analysis
        questions = [record.user_input for record in golden_records]
        answers = [record.reference for record in golden_records]
        contexts_list = [record.reference_contexts for record in golden_records]
        corpus_texts = [record.reference_context for record in corpus_records]
        
        # Analyze question quality
        question_metrics = QualityValidator.analyze_question_quality(questions)
        result.add_statistic("question_quality", question_metrics)
        
        # Check question quality warnings
        if question_metrics.get("short_questions", 0) > len(questions) * 0.1:
            result.add_warning(
                f"High number of short questions: {question_metrics['short_questions']}"
            )
        
        if question_metrics.get("wh_question_ratio", 0) < 0.3:
            result.add_warning(
                f"Low WH-question ratio: {question_metrics['wh_question_ratio']:.2%}"
            )
        
        # Analyze answer quality
        answer_metrics = QualityValidator.analyze_answer_quality(answers)
        result.add_statistic("answer_quality", answer_metrics)
        
        # Check answer quality warnings
        if answer_metrics.get("short_answers", 0) > len(answers) * 0.1:
            result.add_warning(
                f"High number of short answers: {answer_metrics['short_answers']}"
            )
        
        # Analyze context quality
        context_metrics = QualityValidator.analyze_context_quality(contexts_list)
        result.add_statistic("context_quality", context_metrics)
        
        # Check context quality warnings
        if context_metrics.get("empty_contexts", 0) > 0:
            result.add_error(
                f"Found {context_metrics['empty_contexts']} empty contexts"
            )
        
        if context_metrics.get("avg_contexts_per_record", 0) < 2:
            result.add_warning(
                f"Low average contexts per record: {context_metrics['avg_contexts_per_record']:.1f}"
            )
        
        # Compute relevance scores
        relevance_metrics = QualityValidator.compute_relevance_scores(
            questions, answers, contexts_list
        )
        result.add_statistic("relevance_metrics", relevance_metrics)
        
        # Check relevance warnings
        if relevance_metrics.get("avg_qc_overlap", 0) < 0.1:
            result.add_warning(
                f"Low question-context overlap: {relevance_metrics['avg_qc_overlap']:.2%}"
            )
        
        # Corpus analysis
        if corpus_texts:
            corpus_lengths = [len(text) for text in corpus_texts]
            result.add_statistic("corpus_quality", {
                "total_documents": len(corpus_texts),
                "avg_document_length": sum(corpus_lengths) / len(corpus_lengths),
                "min_document_length": min(corpus_lengths),
                "max_document_length": max(corpus_lengths)
            })
        
        return result