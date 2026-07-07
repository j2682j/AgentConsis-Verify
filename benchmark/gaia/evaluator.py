"""
GAIA 閰摯?冽芋蝯?

鞎痊閰摯?箸隞????GAIA ?箸?皜祈岫銝?銵函
"""

from typing import Dict, Any, List, Optional, Union
import time
import re
import json
import csv
from pathlib import Path

from .dataset import GAIADataset
from .metrics import GAIAMetrics
from .answer_matcher import exact_match as gaia_exact_match
from .answer_matcher import partial_match as gaia_partial_match


class GAIAEvaluator:
    """
    鞎痊??evaluation.benchmarks.gaia.evaluator 銝剖?鋆?GAIAEvaluator嚗?鋆?benchmark 閰摯??獢摰??貉?蝞??勗?鞈??渡?瘚???
    
    Args:
        dataset: 甇斗?蝔?閬蝙?函?頛詨鞈???
        level: 甇斗?蝔?閬蝙?函?頛詨鞈???
        local_data_dir: 甇斗?蝔?閬蝙?函?頛詨鞈???
        strict_mode: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        憿?祈澈銝?亙??喳潘?撱箇?撖虫?敺???嗆瘜?雿???瘚???
    
    ??雿:
        ?寞??航?湔?折???撖急?獢?怠??冽????Ｙ??亥?嚗?靘蝙?冽?憓Ⅱ隤?
    """

    def __init__(
        self,
        dataset: Optional[GAIADataset] = None,
        level: Optional[int] = None,
        local_data_dir: Optional[str] = None,
        strict_mode: bool = True
    ):
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? __init__ 瘚?嚗?憪??拐辣???身摰?鞈渲??折???霈?蝥瘜隞交窒?典?銝隞賢銵?銝???
        
        Args:
            dataset: 甇斗?蝔?閬蝙?函?頛詨鞈???
            level: 甇斗?蝔?閬蝙?函?頛詨鞈???
            local_data_dir: 甇斗?蝔?閬蝙?函?頛詨鞈???
            strict_mode: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 ?芣?閮颯?
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        self.dataset = dataset if dataset is not None else GAIADataset(
            level=level,
            local_data_dir=local_data_dir
        )
        self.metrics = GAIAMetrics()
        self.level = level
        self.strict_mode = strict_mode
        
    def evaluate(self, agent: Any, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? evaluate 瘚?嚗?隡?benchmark 隞餃???蝑??蒂?Ｙ???摰???鞈???
        
        Args:
            agent: 甇斗?蝔?閬蝙?函?頛詨鞈???
            max_samples: ?批瑼Ｙ揣?祟?豢?頛詨?賊???澆??詻?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        print("\n[INFO] 開始 GAIA 評估...")
        print(f"   Agent: {getattr(agent, 'name', 'Unknown')}")
        print(f"   等級: {self.level or '全部'}")
        print(f"   評分模式: {'嚴格' if self.strict_mode else '寬鬆'}")

        # 頛鞈???
        dataset = self.dataset.load()
        if not dataset:
            print("   [WARN] 沒有載入任何 GAIA 樣本，無法評估。")
            return self._create_empty_results(agent)

        # ?璅??賊?
        if max_samples:
            dataset = dataset[:max_samples]

        print(f"   評估題數: {len(dataset)}")

        # ?瑁?閰摯
        results = []
        level_stats = {1: {"total": 0, "correct": 0, "partial": 0},
                      2: {"total": 0, "correct": 0, "partial": 0},
                      3: {"total": 0, "correct": 0, "partial": 0}}

        for i, sample in enumerate(dataset):
            if i % 10 == 0:
                print(f"   進度: {i+1}/{len(dataset)}")

            try:
                sample_result = self.evaluate_sample(agent, sample)
                results.append(sample_result)

                # ??蝝絞閮?
                level = sample.get("level", 1)
                if level in level_stats:
                    level_stats[level]["total"] += 1
                    if sample_result["exact_match"]:
                        level_stats[level]["correct"] += 1
                    if sample_result["partial_match"]:
                        level_stats[level]["partial"] += 1

            except Exception as e:
                print(f"   [WARN] 第 {i} 題評估失敗: {e}")
                results.append({
                    "exact_match": False,
                    "partial_match": False,
                    "predicted": None,
                    "expected": sample.get("final_answer"),
                    "error": str(e),
                    "score": 0.0
                })

        # 閮?蝮賡???
        total_samples = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        partial_matches = sum(1 for r in results if r["partial_match"])

        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0.0
        partial_match_rate = partial_matches / total_samples if total_samples > 0 else 0.0

        # 閮?????
        level_metrics = {}
        for level, stats in level_stats.items():
            if stats["total"] > 0:
                level_metrics[f"Level_{level}"] = {
                    "total": stats["total"],
                    "exact_matches": stats["correct"],
                    "partial_matches": stats["partial"],
                    "exact_match_rate": stats["correct"] / stats["total"],
                    "partial_match_rate": stats["partial"] / stats["total"]
                }

        final_results = {
            "benchmark": "GAIA",
            "agent_name": getattr(agent, 'name', 'Unknown'),
            "strict_mode": self.strict_mode,
            "level_filter": self.level,
            "total_samples": total_samples,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "exact_match_rate": exact_match_rate,
            "partial_match_rate": partial_match_rate,
            "level_metrics": level_metrics,
            "detailed_results": results
        }

        print("[OK] GAIA 評估完成")
        print(f"   完全正確率: {exact_match_rate:.2%}")
        print(f"   部分正確率: {partial_match_rate:.2%}")
        for level_name, metrics in level_metrics.items():
            print(f"   {level_name}: 完全正確 {metrics['exact_match_rate']:.2%} / 部分正確 {metrics['partial_match_rate']:.2%}")

        return final_results
    
    def evaluate_sample(self, agent: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? evaluate_sample 瘚?嚗?隡?benchmark 隞餃???蝑??蒂?Ｙ???摰???鞈???
        
        Args:
            agent: 甇斗?蝔?閬蝙?函?頛詨鞈???
            sample: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        response = ""
        predicted_answer = None
        try:
            # 皞?頛詨
            question = sample.get("question", "")
            expected_answer = sample.get("final_answer", "")
            level = sample.get("level", 1)
            task_id = sample.get("task_id", "")

            # 撱箸??內
            prompt = self._build_prompt(question, sample)

            # ?澆?箸隞??
            start_time = time.time()
            if hasattr(agent, "run_sample"):
                response = agent.run_sample(prompt, sample)
            else:
                response = agent.run(prompt)
            execution_time = time.time() - start_time

            # ??蝑?
            predicted_answer = self._extract_answer(response)

            # 閰摯蝑?
            exact_match = self._check_exact_match(predicted_answer, expected_answer)
            partial_match = self._check_partial_match(predicted_answer, expected_answer)

            # 閮??
            if exact_match:
                score = 1.0
            elif partial_match:
                score = 0.5
            else:
                score = 0.0

            return {
                "task_id": task_id,
                "level": level,
                "exact_match": exact_match,
                "partial_match": partial_match,
                "score": score,
                "predicted": predicted_answer,
                "expected": expected_answer,
                "response": response,
                "execution_time": execution_time
            }

        except Exception as e:
            print("   [ERROR] evaluate_sample 執行失敗")
            print(f"   task_id: {sample.get('task_id', '')}")
            print(f"   例外類型: {type(e).__name__}")
            print(f"   例外內容: {e}")
            print(f"   目前預測答案: {predicted_answer!r}")
            print(f"   回應長度: {len(response) if response else 0}")
            if response:
                preview = response[:500].replace("\r", "\\r").replace("\n", "\\n")
                print(f"   回應預覽: {preview}")
            return {
                "task_id": sample.get("task_id", ""),
                "level": sample.get("level", 1),
                "exact_match": False,
                "partial_match": False,
                "score": 0.0,
                "predicted": predicted_answer,
                "expected": sample.get("final_answer", ""),
                "response": response,
                "error": str(e)
            }

    def _create_empty_results(self, agent: Any) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? _create_empty_results 瘚?嚗???GAIAEvaluator ??蝔?瘙???_create_empty_results 撠?????????雿?蝯??Ｙ???
        
        Args:
            agent: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        return {
            "benchmark": "GAIA",
            "agent_name": getattr(agent, 'name', 'Unknown'),
            "strict_mode": self.strict_mode,
            "level_filter": self.level,
            "total_samples": 0,
            "exact_matches": 0,
            "partial_matches": 0,
            "exact_match_rate": 0.0,
            "partial_match_rate": 0.0,
            "level_metrics": {},
            "detailed_results": []
        }

    def _build_prompt(self, question: str, sample: Dict[str, Any]) -> str:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? _build_prompt 瘚?嚗???GAIAEvaluator ??蝔?瘙???_build_prompt 撠?????????雿?蝯??Ｙ???
        
        Args:
            question: ?桀?閬???隞餃???憿??亥岷????
            sample: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        return str(question or "").strip()

    def _extract_answer(self, response: str) -> str:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? _extract_answer 瘚?嚗???GAIAEvaluator ??蝔?瘙???_extract_answer 撠?????????雿?蝯??Ｙ???
        
        Args:
            response: 璅∪???暺?撌亙?Ｙ?????批捆??
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        # 擐??岫??GAIA摰?澆???獢?
        final_answer_pattern = r'FINAL ANSWER:\s*(.+?)(?:\n|$)'
        match = re.search(final_answer_pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # 蝘駁?航??祈?
            answer = answer.strip('[]')
            return answer

        # ??寞?嚗?曉隞?獢?閮?
        answer_patterns = [
            r'蝑?[嚗?]\s*(.+)',
            r'?蝯?獢嚗?]\s*(.+)',
            r'Final answer[嚗?]\s*(.+)',
            r'Answer[嚗?]\s*(.+)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 憒?瘝??曉璅?嚗??單?敺???蝛箄?
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                return line

        return response.strip()

    def _check_exact_match(self, predicted: str, expected: str) -> bool:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? _check_exact_match 瘚?嚗???GAIAEvaluator ??蝔?瘙???_check_exact_match 撠?????????雿?蝯??Ｙ???
        
        Args:
            predicted: 甇斗?蝔?閬蝙?函?頛詨鞈???
            expected: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 bool??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        return gaia_exact_match(predicted, expected)

    def _check_partial_match(self, predicted: str, expected: str) -> bool:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? _check_partial_match 瘚?嚗???GAIAEvaluator ??蝔?瘙???_check_partial_match 撠?????????雿?蝯??Ｙ???
        
        Args:
            predicted: 甇斗?蝔?閬蝙?函?頛詨鞈???
            expected: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 bool??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        return gaia_partial_match(predicted, expected)

    def _normalize_answer(self, answer: str) -> str:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? _normalize_answer 瘚?嚗???GAIAEvaluator ??蝔?瘙???_normalize_answer 撠?????????雿?蝯??Ｙ???
        
        Args:
            answer: 璅∪???暺?撌亙?Ｙ?????批捆??
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not answer:
            return ""

        answer = answer.strip()

        # 瑼Ｘ?臬?舫?????銵?
        if ',' in answer:
            # ??銝行?皞?瘥?蝝?
            parts = [self._normalize_single_answer(p.strip()) for p in answer.split(',')]
            # ??瘥?摨?摨?GAIA閬?嚗?
            parts.sort()
            return ','.join(parts)
        else:
            return self._normalize_single_answer(answer)

    def _normalize_single_answer(self, answer: str) -> str:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? _normalize_single_answer 瘚?嚗???GAIAEvaluator ??蝔?瘙???_normalize_single_answer 撠?????????雿?蝯??Ｙ???
        
        Args:
            answer: 璅∪???暺?撌亙?Ｙ?????批捆??
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 str??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        answer = answer.strip().lower()

        # 蝘駁撣貉???閰?
        articles = ['the', 'a', 'an']
        words = answer.split()
        if words and words[0] in articles:
            words = words[1:]
            answer = ' '.join(words)

        # 蝘駁鞎典馳蝚西????
        answer = answer.replace('$', '').replace('%', '').replace(',', '').replace('£', '')

        # 蝘駁?詨?銝剔?????蝚佗?憒?1,000 -> 1000嚗?
        # 雿????賊?
        answer = re.sub(r'(\d),(\d)', r'\1\2', answer)

        # 蝘駁憭?蝛箸
        answer = ' '.join(answer.split())

        # 蝘駁?怠偏??暺泵??
        answer = answer.rstrip('.,;:!?')

        return answer

    def export_to_gaia_format(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        include_reasoning: bool = True
    ) -> None:
        """
        鞎痊?瑁? GAIAEvaluator 銝剔? export_to_gaia_format 瘚?嚗???GAIAEvaluator ??蝔?瘙???export_to_gaia_format 撠?????????雿?蝯??Ｙ???
        
        Args:
            results: 甇斗?蝔?閬蝙?函?頛詨鞈???
            output_path: 閬???撖怠??獢??桅?頝臬???
            include_reasoning: ?批?臬?甇日?鞈????賣??????????
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 None??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detailed_results = results.get("detailed_results", [])

        with open(output_path, 'w', encoding='utf-8') as f:
            for result in detailed_results:
                gaia_result = {
                    "task_id": result.get("task_id", ""),
                    "model_answer": result.get("predicted", "")
                }

                if include_reasoning:
                    gaia_result["reasoning_trace"] = result.get("response", "")

                f.write(json.dumps(gaia_result, ensure_ascii=False) + '\n')

        print("[OK] GAIA 格式結果已匯出")
        print(f"   輸出檔案: {output_path}")
        print(f"   題數: {len(detailed_results)}")
        print(f"   是否包含 reasoning: {include_reasoning}")

    def export_to_csv(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> None:
        """Export detailed GAIA results as a flat CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_results = results.get("detailed_results", [])
        fieldnames = [
            "task_id",
            "level",
            "exact_match",
            "partial_match",
            "score",
            "predicted",
            "expected",
            "winner_agent_id",
            "execution_time",
            "error",
        ]

        with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in detailed_results:
                writer.writerow({key: result.get(key, "") for key in fieldnames})

        print("[OK] GAIA CSV 結果已匯出")
        print(f"   輸出檔案: {output_path}")
        print(f"   題數: {len(detailed_results)}")


