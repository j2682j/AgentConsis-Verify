"""
定義各Step的資料結構
"""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentConfig:
    """
    建立 SLM_Agent 前的設定資料結構
    """
    agent_id: str
    model_name: str
    temperature: float = 0.3
    confidence_score: float = 0.0
    verifier_scores: list[float] = field(default_factory=list)
    avg_verifier_score: float = 0.0
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
    aggregation_metadata: dict = field(default_factory=dict)
    self_review_metadata: dict = field(default_factory=dict)


@dataclass
class CandidateRun:
    """
    保存單次 Stage1 回答在候選答案選擇階段所需的資訊。

    Args:
     - agent_id: 產生此回答的 Agent ID。
     - run_index: 此回答在 Agent 內的執行序號。
     - answer: 清理後的候選答案。
     - normalized_answer: 用於答案等價分群的正規化文字。
     - reasoning: 與此答案綁定的完整推理文字。

    Returns:
     - CandidateRun: 可追溯至原始 Agent run 的候選成員。
    """

    agent_id: str
    model_name: str
    run_index: int
    answer: str
    normalized_answer: str
    reasoning: str
    agent_confidence: float = 0.0
    agent_answer_frequency: int = 1
    eligible_run_count: int = 1


@dataclass
class AnswerCandidate:
    """
    表示由一個或多個等價 Stage1 回答形成的候選答案群組。

    Args:
     - candidate_key: 候選答案的穩定分群鍵值。
     - representative_answer: 最終輸出與報告使用的代表答案。
     - members: 支持此答案的所有有效 Agent runs。

    Returns:
     - AnswerCandidate: 供 evidence、Versa 與 winner selector 共同評估的候選。
    """

    candidate_key: str
    representative_answer: str
    members: list[CandidateRun] = field(default_factory=list)

    @property
    def supporting_agent_ids(self) -> list[str]:
        return sorted({member.agent_id for member in self.members})

    @property
    def supporting_run_count(self) -> int:
        return len(self.members)


@dataclass
class CandidateEvaluation:
    """
    保存單一答案候選的完整且一致的 winner selection 評估結果。

    Args:
     - candidate_key: 對應的答案候選鍵值。
     - eligible: 候選是否可進入最終選擇。
     - support_tier: 證據支持層級，範圍為 -1 至 4。
     - critical_step_floor: 關鍵推理步驟中最低的 Versa reward probability。
     - critical_step_geometric_mean: 關鍵步驟 reward probabilities 的幾何平均。

    Returns:
     - CandidateEvaluation: 可直接用字典序比較的候選評估紀錄。
    """

    candidate_key: str
    answer: str
    eligible: bool = False
    rejection_reason: str = ""
    support_tier: int = 0
    support_status: str = "no_support"
    direct_support: bool = False
    contradicted: bool = False
    requirement_status: str = "not_available"
    supporting_agent_ids: list[str] = field(default_factory=list)
    supporting_run_count: int = 0
    selected_agent_id: str = ""
    selected_run_index: int = 0
    selected_reasoning: str = ""
    selected_agent_confidence: float = 0.0
    selected_agent_answer_frequency: int = 0
    critical_step_floor: float = 0.0
    critical_step_geometric_mean: float = 0.0
    average_verifier_probability: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEvidenceRecord:
    """
    統一表示 Evidence Prepare 與 Stage1 Tool Use 產生的工具證據。

    Args:
     - tool_name: 產生結果的工具或 handler 名稱。
     - output_type: final_answer、intermediate_value、evidence_text 或 failed。
     - value: 可供比對的答案或中間值。
     - role: 工具結果在解題流程中的語意角色。
     - trusted: 結果是否通過既有的 validation / trust gate。
     - evidence_valid: 結果是否包含可使用的證據。
     - source_scope: evidence_prepare 或 stage1_tool_use。

    Returns:
     - ToolEvidenceRecord: 可供 EvidenceSupportChecker 使用的標準工具紀錄。
    """

    tool_name: str
    output_type: str
    value: str = ""
    role: str = ""
    trusted: bool = False
    evidence_valid: bool = False
    source_scope: str = ""
    agent_id: str = ""
    run_index: int = 0
    status: str = ""
    evidence_text: str = ""
    missing_inputs: list[str] = field(default_factory=list)
    next_action_hint: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepSupportResult:
    """
    記錄單一推理步驟與工具證據之間的關係。

    Args:
     - step_index: 推理步驟編號。
     - step_text: 已解析的推理步驟文字。
     - status: supported、unsupported、contradicted 或 tool_failed。
     - matched_tool_values: 此步驟實際使用的工具答案或中間值。
     - source_tools: 支持或衝突來源的工具名稱。

    Returns:
     - StepSupportResult: 可與 Versa reward probability 合併輸出的步驟結果。
    """

    step_index: int
    step_text: str
    status: str
    matched_tool_values: list[str] = field(default_factory=list)
    source_tools: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class AgentEvidenceSupportSummary:
    """
    彙整一個 Agent 的最終答案與推理步驟是否受到工具證據支持。

    Args:
     - agent_id: 被檢查的 Agent。
     - status: Agent 層級的證據支持狀態。
     - priority: winner selection 使用的離散優先層級。
     - step_results: 每個 reasoning step 的支持結果。
     - evidence_records: 實際參與本次判斷的標準工具紀錄。

    Returns:
     - AgentEvidenceSupportSummary: Stage2 與 winner selection 共用的支持摘要。
    """

    agent_id: str
    status: str
    priority: int
    step_results: list[StepSupportResult] = field(default_factory=list)
    evidence_records: list[ToolEvidenceRecord] = field(default_factory=list)
    matched_final_values: list[str] = field(default_factory=list)
    trusted_final_answers: list[str] = field(default_factory=list)
    tool_failure_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierScoreByReasoning:
    """
    Judge 根據 Agent 的 Reasoning 給分的資料結構
    """
    verifier_id: str
    target_agent_id: str
    verifier_score: float  # VersaPRM average reward probability, 0.0 ~ 1.0
    step_scores: list[dict] = field(default_factory=list)
    raw_reply: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


JudgeScoreByReasoning = VerifierScoreByReasoning

@dataclass
class NetworkSummary:
    """
    整次任務的最終輸出資料結構
    """
    question: str
    final_answer: str
    winner_agent_id: str
    stage1_results: list[AgentReasoningSummary]
    verifier_results: list[VerifierScoreByReasoning]
    agent_scores: list[AgentConfig]
    metadata: dict
