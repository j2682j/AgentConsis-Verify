"""
定義各Step的資料結構
"""
from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    """
    建立 SLM_Agent 前的設定資料結構
    """
    agent_id: str
    model_name: str
    temperature: float = 0.5
    confidence_score: float = 0.0
    judge_scores: list[float] = field(default_factory=list)
    avg_judge_score: float = 0.0
    penalty_score: float = 0.0
    penalty_reasons: list[str] = field(default_factory=list)
    total_score: float = 0.0
    
    
@dataclass
class EachAgentReply:
    """
    單一 Agent 的單次回答
    """
    agent_id: str
    model_name: str
    run_index: int
    raw_reply: str
    reasoning: str
    final_answer: str
    parse_completed: bool
    tool_context: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)
    structured_output: dict = field(default_factory=dict)
    schema_valid: bool = False
    schema_errors: list[str] = field(default_factory=list)
    repair_applied: bool = False
    repair_actions: list[str] = field(default_factory=list)
    eligible_for_winner: bool = True
    validity_labels: list[str] = field(default_factory=list)
    final_answer_source: str = "original"
    repair_metadata: dict = field(default_factory=dict)
    context_budget: dict = field(default_factory=dict)

    
@dataclass
class AgentReasoningSummary:
    """
    單一 Agent 跑完三次後的整理結果
    """
    agent_id: str
    model_name: str
    runs: list[EachAgentReply]
    compressed_answer: str
    compressed_reasoning: str
    confidence_score: float
    active: bool
    valid_run_count: int = 0
    invalid_run_count: int = 0
    abstention_run_count: int = 0
    eligible_run_count: int = 0
    run_validity_labels: list[str] = field(default_factory=list)
    winner_selection_eligible: bool = True
    winner_selection_status: str = "answerable"

@dataclass
class JudgeScoreByReasoning:
    """
    Judge 根據 Agent 的 Reasoning 給分的資料結構
    """
    judge_agent_id: str
    target_agent_id: str
    judge_score: float  # -5 ~ 5
    step_scores: list[dict] = field(default_factory=list)
    raw_reply: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class NetworkSummary:
    """
    整次任務的最終輸出資料結構
    """
    question: str
    final_answer: str
    winner_agent_id: str
    stage1_results: list[AgentReasoningSummary]
    judge_results: list[JudgeScoreByReasoning]
    agent_scores: list[AgentConfig]
    metadata: dict
