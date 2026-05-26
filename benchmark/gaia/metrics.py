"""
GAIA 閰摯??璅∠?

閮? GAIA ?賊???隡唳?璅?
"""

from typing import Dict, Any, List, Optional
import numpy as np


class GAIAMetrics:
    """
    鞎痊??evaluation.benchmarks.gaia.metrics 銝剖?鋆?GAIAMetrics嚗?鋆?benchmark 閰摯??獢摰??貉?蝞??勗?鞈??渡?瘚???
    
    Args:
        ?⊥?蝣箏遣瑽??賂??航?? dataclass 甈???閮剖澆遣蝡隞嗚?
    
    Returns:
        憿?祈澈銝?亙??喳潘?撱箇?撖虫?敺???嗆瘜?雿???瘚???
    
    ??雿:
        ?寞??航?湔?折???撖急?獢?怠??冽????Ｙ??亥?嚗?靘蝙?冽?憓Ⅱ隤?
    """

    @staticmethod
    def calculate_exact_match_rate(results: List[Dict[str, Any]]) -> float:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? calculate_exact_match_rate 瘚?嚗???GAIAMetrics ??蝔?瘙???calculate_exact_match_rate 撠?????????雿?蝯??Ｙ???
        
        Args:
            results: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 float??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not results:
            return 0.0

        exact_matches = sum(1 for r in results if r.get("exact_match", False))
        return exact_matches / len(results)

    @staticmethod
    def calculate_partial_match_rate(results: List[Dict[str, Any]]) -> float:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? calculate_partial_match_rate 瘚?嚗???GAIAMetrics ??蝔?瘙???calculate_partial_match_rate 撠?????????雿?蝯??Ｙ???
        
        Args:
            results: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 float??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not results:
            return 0.0

        partial_matches = sum(1 for r in results if r.get("partial_match", False))
        return partial_matches / len(results)

    @staticmethod
    def calculate_level_metrics(
        results: List[Dict[str, Any]],
        level: int
    ) -> Dict[str, float]:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? calculate_level_metrics 瘚?嚗???GAIAMetrics ??蝔?瘙???calculate_level_metrics 撠?????????雿?蝯??Ｙ???
        
        Args:
            results: 甇斗?蝔?閬蝙?函?頛詨鞈???
            level: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, float]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        level_results = [r for r in results if r.get("level") == level]

        if not level_results:
            return {
                "total": 0,
                "exact_match_rate": 0.0,
                "partial_match_rate": 0.0,
                "average_score": 0.0
            }

        exact_matches = sum(1 for r in level_results if r.get("exact_match", False))
        partial_matches = sum(1 for r in level_results if r.get("partial_match", False))
        scores = [r.get("score", 0.0) for r in level_results]

        return {
            "total": len(level_results),
            "exact_match_rate": exact_matches / len(level_results),
            "partial_match_rate": partial_matches / len(level_results),
            "average_score": sum(scores) / len(scores) if scores else 0.0
        }

    @staticmethod
    def calculate_average_execution_time(results: List[Dict[str, Any]]) -> float:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? calculate_average_execution_time 瘚?嚗???GAIAMetrics ??蝔?瘙???calculate_average_execution_time 撠?????????雿?蝯??Ｙ???
        
        Args:
            results: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 float??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        execution_times = [r.get("execution_time", 0.0) for r in results if "execution_time" in r]
        return sum(execution_times) / len(execution_times) if execution_times else 0.0

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? compute_metrics 瘚?嚗???GAIAMetrics ??蝔?瘙???compute_metrics 撠?????????雿?蝯??Ｙ???
        
        Args:
            results: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not results:
            return self._empty_metrics()

        # ?箇???
        total = len(results)
        exact_match_rate = self.calculate_exact_match_rate(results)
        partial_match_rate = self.calculate_partial_match_rate(results)
        avg_execution_time = self.calculate_average_execution_time(results)

        # ????
        level_metrics = {
            "Level_1": self.calculate_level_metrics(results, 1),
            "Level_2": self.calculate_level_metrics(results, 2),
            "Level_3": self.calculate_level_metrics(results, 3)
        }

        # ?蝯梯?
        scores = [r.get("score", 0.0) for r in results]
        score_stats = self._compute_score_statistics(scores)

        # ?扯??
        performance_analysis = self._analyze_performance(results)

        return {
            "total_samples": total,
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "average_execution_time": avg_execution_time,
            "level_metrics": level_metrics,
            "score_statistics": score_stats,
            "performance_analysis": performance_analysis
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? _empty_metrics 瘚?嚗???GAIAMetrics ??蝔?瘙???_empty_metrics 撠?????????雿?蝯??Ｙ???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        return {
            "total_samples": 0,
            "exact_match_rate": 0.0,
            "partial_match_rate": 0.0,
            "average_execution_time": 0.0,
            "level_metrics": {
                "Level_1": {"total": 0, "exact_match_rate": 0.0, "partial_match_rate": 0.0, "average_score": 0.0},
                "Level_2": {"total": 0, "exact_match_rate": 0.0, "partial_match_rate": 0.0, "average_score": 0.0},
                "Level_3": {"total": 0, "exact_match_rate": 0.0, "partial_match_rate": 0.0, "average_score": 0.0}
            },
            "score_statistics": {},
            "performance_analysis": {}
        }

    def _compute_score_statistics(self, scores: List[float]) -> Dict[str, float]:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? _compute_score_statistics 瘚?嚗???GAIAMetrics ??蝔?瘙???_compute_score_statistics 撠?????????雿?蝯??Ｙ???
        
        Args:
            scores: 閰摯???撌亙?瑁?敺??蝯????貉???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, float]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not scores:
            return {}

        return {
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores),
            "min": min(scores),
            "max": max(scores),
            "q1": np.percentile(scores, 25),
            "q3": np.percentile(scores, 75)
        }

    def _analyze_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? _analyze_performance 瘚?嚗???GAIAMetrics ??蝔?瘙???_analyze_performance 撠?????????雿?蝯??Ｙ???
        
        Args:
            results: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not results:
            return {}

        # ??蝝?蝯???
        level_performance = {}
        for level in [1, 2, 3]:
            level_results = [r for r in results if r.get("level") == level]
            if level_results:
                exact_matches = sum(1 for r in level_results if r.get("exact_match", False))
                level_performance[f"Level_{level}"] = {
                    "sample_count": len(level_results),
                    "success_count": exact_matches,
                    "success_rate": exact_matches / len(level_results)
                }

        # 閮???漲?脰”??
        difficulty_progression = self._analyze_difficulty_progression(level_performance)

        # ?航炊??
        error_analysis = self._analyze_errors(results)

        return {
            "level_performance": level_performance,
            "difficulty_progression": difficulty_progression,
            "error_analysis": error_analysis
        }

    def _analyze_difficulty_progression(self, level_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? _analyze_difficulty_progression 瘚?嚗???GAIAMetrics ??蝔?瘙???_analyze_difficulty_progression 撠?????????雿?蝯??Ｙ???
        
        Args:
            level_performance: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        progression = {}

        levels = ["Level_1", "Level_2", "Level_3"]
        for i in range(len(levels) - 1):
            current_level = levels[i]
            next_level = levels[i + 1]

            if current_level in level_performance and next_level in level_performance:
                current_rate = level_performance[current_level]["success_rate"]
                next_rate = level_performance[next_level]["success_rate"]

                progression[f"{current_level}_to_{next_level}"] = {
                    "drop_rate": current_rate - next_rate,
                    "relative_drop": (current_rate - next_rate) / current_rate if current_rate > 0 else 0
                }

        return progression

    def _analyze_errors(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? _analyze_errors 瘚?嚗???GAIAMetrics ??蝔?瘙???_analyze_errors 撠?????????雿?蝯??Ｙ???
        
        Args:
            results: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        total_errors = sum(1 for r in results if not r.get("exact_match", False))
        partial_correct = sum(1 for r in results if r.get("partial_match", False) and not r.get("exact_match", False))
        complete_wrong = sum(1 for r in results if not r.get("partial_match", False) and not r.get("exact_match", False))

        return {
            "total_errors": total_errors,
            "partial_correct": partial_correct,
            "complete_wrong": complete_wrong,
            "error_rate": total_errors / len(results) if results else 0,
            "partial_correct_rate": partial_correct / total_errors if total_errors > 0 else 0
        }

    @staticmethod
    def compare_results(results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAMetrics 銝剔? compare_results 瘚?嚗???GAIAMetrics ??蝔?瘙???compare_results 撠?????????雿?蝯??Ｙ???
        
        Args:
            results1: 甇斗?蝔?閬蝙?函?頛詨鞈???
            results2: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        comparison = {
            "exact_match_rate_diff": results1.get("exact_match_rate", 0) - results2.get("exact_match_rate", 0),
            "partial_match_rate_diff": results1.get("partial_match_rate", 0) - results2.get("partial_match_rate", 0),
            "execution_time_diff": results1.get("average_execution_time", 0) - results2.get("average_execution_time", 0)
        }

        # ??蝝?頛?
        level_comparison = {}
        for level in ["Level_1", "Level_2", "Level_3"]:
            if level in results1.get("level_metrics", {}) and level in results2.get("level_metrics", {}):
                level1 = results1["level_metrics"][level]
                level2 = results2["level_metrics"][level]
                level_comparison[level] = {
                    "exact_match_rate_diff": level1.get("exact_match_rate", 0) - level2.get("exact_match_rate", 0),
                    "score_diff": level1.get("average_score", 0) - level2.get("average_score", 0)
                }

        comparison["level_comparison"] = level_comparison

        return comparison

 
