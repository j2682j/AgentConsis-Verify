"""
GAIA 鞈????交芋蝯?

鞎痊敺?HuggingFace 頛 GAIA (General AI Assistants) 鞈???
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data"


class GAIADataset:
    """
    鞎痊??evaluation.benchmarks.gaia.dataset 銝剖?鋆?GAIADataset嚗?鋆?benchmark 閰摯??獢摰??貉?蝞??勗?鞈??渡?瘚???
    
    Args:
        dataset_name: 甇斗?蝔?閬蝙?函?頛詨鞈???
        split: 甇斗?蝔?閬蝙?函?頛詨鞈???
        level: 甇斗?蝔?閬蝙?函?頛詨鞈???
        local_data_dir: 甇斗?蝔?閬蝙?函?頛詨鞈???
    
    Returns:
        憿?祈澈銝?亙??喳潘?撱箇?撖虫?敺???嗆瘜?雿???瘚???
    
    ??雿:
        ?寞??航?湔?折???撖急?獢?怠??冽????Ｙ??亥?嚗?靘蝙?冽?憓Ⅱ隤?
    """

    def __init__(
        self,
        dataset_name: str = "gaia-benchmark/GAIA",
        split: str = "validation",
        level: Optional[int] = None,
        local_data_dir: Optional[Union[str, Path]] = None
    ):
        """
        鞎痊?瑁? GAIADataset 銝剔? __init__ 瘚?嚗?憪??拐辣???身摰?鞈渲??折???霈?蝥瘜隞交窒?典?銝隞賢銵?銝???
        
        Args:
            dataset_name: 甇斗?蝔?閬蝙?函?頛詨鞈???
            split: 甇斗?蝔?閬蝙?函?頛詨鞈???
            level: 甇斗?蝔?閬蝙?函?頛詨鞈???
            local_data_dir: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 ?芣?閮颯?
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        self.dataset_name = dataset_name
        self.split = split
        self.level = level
        self.local_data_dir = Path(local_data_dir) if local_data_dir else None
        self.data = []
        self._is_local = self._check_if_local_source()

    def _check_if_local_source(self) -> bool:
        """
        鞎痊?瑁? GAIADataset 銝剔? _check_if_local_source 瘚?嚗???GAIADataset ??蝔?瘙???_check_if_local_source 撠?????????雿?蝯??Ｙ???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 bool??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if self.local_data_dir and self.local_data_dir.exists():
            return True
        return False

    def load(self) -> List[Dict[str, Any]]:
        """
        鞎痊?瑁? GAIADataset 銝剔? load 瘚?嚗???唳?憭鞈?靘?銝西???蝟餌絞?航????澆???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 List[Dict[str, Any]]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if self.data:
            return self.data

        if self._is_local:
            self.data = self._load_from_local()
        else:
            self.data = self._load_from_huggingface()

        # ??蝝?瞈?
        # if self.level is not None:
        #     self.data = [item for item in self.data if item.get("level") == self.level]

        print("[OK] GAIA 資料集載入完成")
        print(f"   來源: {self.dataset_name}")
        print(f"   分割: {self.split}")
        print(f"   等級: {self.level or '全部'}")
        print(f"   題數: {len(self.data)}")

        return self.data

    def _load_from_local(self) -> List[Dict[str, Any]]:
        """
        鞎痊?瑁? GAIADataset 銝剔? _load_from_local 瘚?嚗???GAIADataset ??蝔?瘙???_load_from_local 撠?????????雿?蝯??Ｙ???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 List[Dict[str, Any]]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        data = []

        if not self.local_data_dir or not self.local_data_dir.exists():
            print("   [WARN] 找不到本機 GAIA 資料目錄。")
            return data

        try:
            import pandas as pd
        except ImportError:
            print("   [WARN] 缺少 pandas 或 pyarrow，無法讀取 parquet。")
            return data

        metadata_dir = self.local_data_dir / "2023" / self.split
        if not metadata_dir.exists():
            print(f"   [WARN] 找不到 GAIA metadata 目錄: {metadata_dir}")
            return data

        candidate_files = []
        if self.level is not None:
            candidate_files.append(metadata_dir / f"metadata.level{self.level}.parquet")
        candidate_files.append(metadata_dir / "metadata.parquet")

        metadata_file = next((path for path in candidate_files if path.exists()), None)
        if metadata_file is None:
            expected = ", ".join(str(path) for path in candidate_files)
            print(f"   [WARN] 找不到 metadata 檔案: {expected}")
            return data

        try:
            df = pd.read_parquet(metadata_file)
            records = df.to_dict(orient="records")

            for item in records:
                if item.get("task_id") == "0-0-0-0-0":
                    continue

                if item.get("file_name"):
                    item["file_name"] = str(metadata_dir / item["file_name"])

                data.append(self._standardize_item(item))

            print(f"   已讀取 metadata: {metadata_file.name} ({len(data)} 題)")
        except Exception as e:
            print(f"   [WARN] 讀取 metadata 失敗: {metadata_file.name} - {e}")

        return data

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """
        鞎痊?瑁? GAIADataset 銝剔? _load_from_huggingface 瘚?嚗???GAIADataset ??蝔?瘙???_load_from_huggingface 撠?????????雿?蝯??Ｙ???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 List[Dict[str, Any]]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        try:
            from huggingface_hub import snapshot_download
            import os
            import pandas as pd
            from pathlib import Path

            print(f"   從 HuggingFace 下載資料集: {self.dataset_name}")

            # ??HF token
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("   [WARN] 找不到 HF_TOKEN 環境變數，無法從 HuggingFace 下載。")
                print("   請在 .env 設定: HF_TOKEN=your_token")
                return []

            # 銝?鞈???砍
            print("正在下載 GAIA 資料集...")
            # Store downloaded GAIA data in the repository data directory.
            local_dir = EXTERNAL_DATA_DIR / "gaia"
            local_dir.mkdir(parents=True, exist_ok=True)

            try:
                snapshot_download(
                    repo_id=self.dataset_name,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    token=hf_token,
                    local_dir_use_symlinks=False  # Windows?詨捆??
                )
                print(f"   資料集下載完成: {local_dir}")
            except Exception as e:
                print(f"   [WARN] 下載失敗: {e}")
                print("   請確認 HuggingFace 權限與網路連線。")
                return []

            metadata_dir = local_dir / "2023" / self.split
            candidate_files = []
            if self.level is not None:
                candidate_files.append(metadata_dir / f"metadata.level{self.level}.parquet")
            candidate_files.append(metadata_dir / "metadata.parquet")

            metadata_file = next((path for path in candidate_files if path.exists()), None)
            if metadata_file is None:
                expected = ", ".join(str(path) for path in candidate_files)
                print(f"   [WARN] 找不到 metadata 檔案: {expected}")
                return []

            df = pd.read_parquet(metadata_file)
            records = df.to_dict(orient="records")

            data = []
            for item in records:
                if item.get("task_id") == "0-0-0-0-0":
                    continue

                if item.get("file_name"):
                    item["file_name"] = str(metadata_dir / item["file_name"])

                standardized_item = self._standardize_item(item)
                data.append(standardized_item)

            print(f"   已載入 HuggingFace GAIA metadata: {metadata_file.name} ({len(data)} 題)")
            return data

        except ImportError:
            print("   [WARN] 缺少 huggingface_hub、pandas 或 pyarrow。")
            print("   請安裝: pip install huggingface_hub pandas pyarrow")
            return []
        except Exception as e:
            print(f"   [WARN] 載入 GAIA 資料失敗: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _standardize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIADataset 銝剔? _standardize_item 瘚?嚗???GAIADataset ??蝔?瘙???_standardize_item 撠?????????雿?蝯??Ｙ???
        
        Args:
            item: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        # GAIA鞈???璅?摮挾
        standardized = {
            "task_id": item.get("task_id", ""),
            "question": item.get("Question", item.get("question", "")),
            "level": item.get("Level", item.get("level", "1")),
            "final_answer": item.get("Final answer", item.get("final_answer", "")),
            "file_name": item.get("file_name", ""),
            "file_path": item.get("file_path", ""),
            "annotator_metadata": item.get("Annotator Metadata", item.get("annotator_metadata", {})),
            "steps": item.get("Steps", item.get("steps", 0)),
            "tools": item.get("Tools", item.get("tools", [])),
            "raw_item": item  # 靽???鞈?
        }

        return standardized
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIADataset 銝剔? get_sample 瘚?嚗???GAIADataset ??蝔?瘙???get_sample 撠?????????雿?蝯??Ｙ???
        
        Args:
            index: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not self.data:
            self.load()
        return self.data[index] if index < len(self.data) else {}

    def get_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        鞎痊?瑁? GAIADataset 銝剔? get_by_level 瘚?嚗???GAIADataset ??蝔?瘙???get_by_level 撠?????????雿?蝯??Ｙ???
        
        Args:
            level: 甇斗?蝔?閬蝙?函?頛詨鞈???
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 List[Dict[str, Any]]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not self.data:
            self.load()
        return [item for item in self.data if item.get("level") == level]

    def get_level_distribution(self) -> Dict[int, int]:
        """
        鞎痊?瑁? GAIADataset 銝剔? get_level_distribution 瘚?嚗???GAIADataset ??蝔?瘙???get_level_distribution 撠?????????雿?蝯??Ｙ???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[int, int]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not self.data:
            self.load()

        distribution = {1: 0, 2: 0, 3: 0}
        for item in self.data:
            level = item.get("level", 1)
            if level in distribution:
                distribution[level] += 1

        return distribution

    def get_statistics(self) -> Dict[str, Any]:
        """
        鞎痊?瑁? GAIADataset 銝剔? get_statistics 瘚?嚗???GAIADataset ??蝔?瘙???get_statistics 撠?????????雿?蝯??Ｙ???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 Dict[str, Any]??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not self.data:
            self.load()

        level_dist = self.get_level_distribution()

        # 蝯梯??閬?獢?璅???
        with_files = sum(1 for item in self.data if item.get("file_name"))

        # 蝯梯?撟喳?甇交
        steps_list = [item.get("steps", 0) for item in self.data if item.get("steps")]
        avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0

        return {
            "total_samples": len(self.data),
            "level_distribution": level_dist,
            "samples_with_files": with_files,
            "average_steps": avg_steps,
            "split": self.split
        }

    def __len__(self) -> int:
        """
        鞎痊?瑁? GAIADataset 銝剔? __len__ 瘚?嚗???GAIADataset ??蝔?瘙???__len__ 撠?????????雿?蝯??Ｙ???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 int??
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not self.data:
            self.load()
        return len(self.data)

    def __iter__(self):
        """
        鞎痊?瑁? GAIADataset 銝剔? __iter__ 瘚?嚗???GAIADataset ??蝔?瘙???__iter__ 撠?????????雿?蝯??Ｙ???
        
        Args:
            ?～?
        
        Returns:
            ?瑁?蝯?嚗?賢?璅酉??嚗????亦 ?芣?閮颯?
        
        ??雿:
            ?航霈???湔?拐辣???獢??冽????亥?嚗?靘?怠?舐Ⅱ隤雿??
        """
        if not self.data:
            self.load()
        return iter(self.data)

 

