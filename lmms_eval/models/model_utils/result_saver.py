"""
Utility module for incrementally saving per-sample evaluation results.

This module provides functionality to save results as they are generated during
evaluation, enabling detailed error analysis and preventing data loss if
evaluation crashes.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger as eval_logger


class IncrementalResultSaver:
    """
    Saves per-sample results to a JSONL file incrementally during evaluation.
    
    This class handles thread-safe (or rank-safe in distributed settings)
    writing of per-sample results as evaluation progresses.
    """
    
    def __init__(self, output_dir: str, task_name: str, rank: int = 0):
        """
        Initialize the result saver.
        
        Args:
            output_dir: Directory where results will be saved
            task_name: Name of the task being evaluated
            rank: Rank of the current process (for distributed evaluation)
        """
        self.output_dir = output_dir
        self.task_name = task_name
        self.rank = rank
        
        # Only write on rank 0 to avoid conflicts in distributed evaluation
        self.should_write = (rank == 0)
        
        if self.should_write:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Determine output file path
            self.output_file = os.path.join(
                output_dir, 
                f"per_sample_results_{task_name}.jsonl"
            )
            
            eval_logger.info(f"Per-sample results will be saved to: {self.output_file}")
    
    def save_result(self, result: Dict[str, Any]) -> bool:
        """
        Save a single result to the JSONL file.
        
        Args:
            result: Dictionary containing the result data
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.should_write:
            return True
        
        try:
            with open(self.output_file, 'a') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            return True
        except Exception as e:
            eval_logger.error(f"Failed to save result: {e}")
            return False
    
    def get_output_file(self) -> Optional[str]:
        """Get the path to the output file."""
        return self.output_file if self.should_write else None


def extract_videomme_result_data(doc: Dict[str, Any], metrics: Dict[str, Any], 
                                  filtered_responses: list, doc_id: Any) -> Dict[str, Any]:
    """
    Extract relevant data from videomme evaluation for per-sample result saving.
    
    This function converts the raw evaluation data (document, metrics, responses)
    into a structured format suitable for analysis and archival.
    
    Args:
        doc: The document/sample from the dataset
        metrics: The metrics dictionary returned by task.process_results()
        filtered_responses: List of filtered model responses
        doc_id: The document ID
        
    Returns:
        Dictionary containing:
        - question_id: ID of the question
        - video_id: ID of the video
        - video_path: Path to the video file (if available)
        - question: The question text
        - options: List of answer options
        - ground_truth_answer: The correct answer (A/B/C/D)
        - model_answer: The answer predicted by the model (A/B/C/D)
        - is_correct: Whether the model's answer matches the ground truth
        - metadata: Additional metadata (duration, category, etc.)
        - raw_model_response: The raw response from the model (for debugging)
    """
    # Extract the main metric data (videomme_perception_score contains all info)
    metric_data = metrics.get("videomme_perception_score", {})
    
    # Determine correctness
    is_correct = (
        metric_data.get("pred_answer") == metric_data.get("answer")
        if metric_data.get("pred_answer") and metric_data.get("answer")
        else None
    )
    
    result = {
        "question_id": doc.get("question_id"),
        "video_id": doc.get("videoID"),
        "video_path": metric_data.get("video_path"),
        "question": doc.get("question"),
        "options": doc.get("options", []),
        "ground_truth_answer": doc.get("answer"),
        "model_answer": metric_data.get("pred_answer"),
        "is_correct": is_correct,
        "metadata": {
            "duration": metric_data.get("duration"),
            "domain": metric_data.get("category"),
            "sub_category": metric_data.get("sub_category"),
            "task_type": metric_data.get("task_category"),
        },
        "raw_model_response": filtered_responses[0] if filtered_responses else None,
        "doc_id": str(doc_id),
    }
    
    return result


def extract_generic_result_data(doc: Dict[str, Any], metrics: Dict[str, Any],
                                filtered_responses: list, doc_id: Any) -> Dict[str, Any]:
    """
    Extract relevant data from generic evaluation for per-sample result saving.
    
    This is a more general version that works for any task type.
    
    Args:
        doc: The document/sample from the dataset
        metrics: The metrics dictionary returned by task.process_results()
        filtered_responses: List of filtered model responses
        doc_id: The document ID
        
    Returns:
        Dictionary containing general result information
    """
    result = {
        "doc_id": str(doc_id),
        "metrics": metrics,
        "raw_model_response": filtered_responses[0] if filtered_responses else None,
    }
    
    # Try to add common fields if they exist
    for field in ["question_id", "id", "qid"]:
        if field in doc:
            result["question_id"] = doc[field]
            break
    
    if "question" in doc:
        result["question"] = doc["question"]
    
    if "answer" in doc:
        result["ground_truth_answer"] = doc["answer"]
    
    return result
