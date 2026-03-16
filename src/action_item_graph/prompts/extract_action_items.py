"""
Action item extraction prompts and response models.

Uses OpenAI structured output for deterministic extraction.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Valid status values that match ActionItemStatus enum
ImpliedStatusType = Literal['completed', 'in_progress', 'cancelled', 'deferred']


# =============================================================================
# Topic Extraction Model
# =============================================================================


class ExtractedTopic(BaseModel):
    """High-level topic/theme that an action item belongs to."""

    name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description='High-level topic name in 3-5 words. Should capture the strategic initiative '
        'or project theme. Examples: "Q1 Sales Expansion", "Annual Security Compliance", '
        '"Website Redesign Project". Be consistent: related action items should share the SAME '
        'topic name.',
    )
    context: str = Field(
        ...,
        max_length=200,
        description='Brief explanation (1-2 sentences) of why the action item belongs to this '
        'topic and what the topic encompasses.',
    )


# =============================================================================
# Response Models for Structured Output
# =============================================================================


class ExtractedActionItem(BaseModel):
    """A single action item extracted from the transcript."""

    action_item_text: str = Field(
        ...,
        description='The verbatim or near-verbatim action item text from the transcript. '
        'Include enough context to understand what needs to be done.',
    )
    owner: str = Field(
        ...,
        description='The person responsible for this action item, as stated in the conversation. '
        'Use the name exactly as mentioned (e.g., "Sarah", "John", "the client").',
    )
    summary: str = Field(
        ...,
        description='A concise 1-sentence summary of what needs to be done. '
        'Should be actionable and clear even without the full transcript context.',
    )
    conversation_context: str = Field(
        ...,
        description='1-2 sentences of surrounding conversation context that helps clarify '
        'the action item (e.g., why it matters, what it relates to).',
    )
    topic: ExtractedTopic = Field(
        ...,
        description='The high-level topic/project this action item belongs to. '
        'Use 3-5 words that capture the strategic initiative.',
    )
    due_date_text: str | None = Field(
        default=None,
        description='The due date or timeframe if mentioned (e.g., "by Friday", "next week", '
        '"end of month"). Null if no timeline mentioned.',
    )
    is_status_update: bool = Field(
        default=False,
        description='True if this is a STATUS UPDATE about a previously committed item '
        '(e.g., "I sent that deck", "Done!", "The proposal is finished"). '
        'False if this is a NEW action item commitment.',
    )
    implied_status: ImpliedStatusType | None = Field(
        default=None,
        description='For status updates only: what status does this imply? '
        'Must be one of: "completed", "in_progress", "cancelled", "deferred". Null for new items.',
    )
    owner_type: Literal["named", "role_inferred", "unconfirmed"] = Field(
        default="named",
        description='How the owner was identified: "named" (real name from context), '
        '"role_inferred" (role from context, e.g. "the account executive"), '
        '"unconfirmed" (could not identify speaker). '
        'NEVER use bare diarization labels (A, B, C, Speaker 1) as owner.',
    )
    is_user_owned: bool = Field(
        default=False,
        description='True if this action item belongs to the recording user '
        '(the person who initiated the meeting/recording). '
        'Only set to true if the user can be confidently identified.',
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description='Confidence score (0.0 to 1.0) for this extraction. '
        'Lower if ambiguous or uncertain.',
    )

    # Five-Field Commitment Framework (Phase 1: Quality Overhaul)
    commitment_strength: Literal['explicit', 'conditional', 'weak', 'observation'] = Field(
        default='explicit',
        description='How strongly the speaker committed to this action: '
        '"explicit" = clear first-person commitment ("I will...", "Let me..."), '
        '"conditional" = depends on something ("If X happens, I\'ll..."), '
        '"weak" = vague intention ("We should...", "It would be nice if..."), '
        '"observation" = third-party expectation, NOT a personal commitment.',
    )
    decision_context: str | None = Field(
        default=None,
        max_length=200,
        description='What was decided or agreed upon that makes this action item necessary. '
        'Null if no explicit decision was made.',
    )
    definition_of_done: str | None = Field(
        default=None,
        max_length=200,
        description='What proves this action item is complete '
        '(e.g., "Proposal email received by client"). Null if not determinable.',
    )

    # Scoring dimensions (Phase 1: Quality Overhaul)
    score_impact: int = Field(
        default=3, ge=1, le=5,
        description='Business impact (1=trivial administrative, 5=deal-critical or revenue-blocking).',
    )
    score_urgency: int = Field(
        default=3, ge=1, le=5,
        description='Time sensitivity (1=no rush, 5=blocking/overdue right now).',
    )
    score_specificity: int = Field(
        default=3, ge=1, le=5,
        description='How actionable and specific (1=vague "follow up", 5=crystal clear deliverable).',
    )
    score_effort: int = Field(
        default=3, ge=1, le=5,
        description='Estimated effort (1=5 minutes, 5=multi-day project).',
    )


class ExtractionResult(BaseModel):
    """Complete extraction result from a transcript."""

    action_items: list[ExtractedActionItem] = Field(
        default_factory=list,
        description='List of all action items extracted from the transcript.',
    )
    has_action_items: bool = Field(
        ...,
        description='True if any action items were found, False otherwise.',
    )
    extraction_notes: str | None = Field(
        default=None,
        description='Optional notes about the extraction (e.g., "Transcript appears to be '
        'a casual conversation with no clear commitments").',
    )


# =============================================================================
# Prompt Templates
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert at identifying actionable commitments from sales call transcripts and meeting notes.

## YOUR TASK — Two-Step Focused Extraction

Follow this two-step process to identify ONLY genuine, actionable commitments:

### Step 1: Identify Commitment Signals

Scan the transcript for COMMITMENT LANGUAGE — moments where a specific person pledges to do something concrete.

<commitment_signals>
- First-person future tense: "I will...", "I'll...", "Let me..."
- Explicit promises: "I can have that to you by..."
- Acceptance of assignment: "I'll take care of that", "Leave that with me"
- Scheduling commitments: "Let's schedule...", "I'll set up a time..."
- Deadline commitments: "I'll have it done by Friday"
</commitment_signals>

<not_commitments>
- Observations about others: "Salesforce numbers should be fixed by end of week" (who committed?)
- Vague intentions: "We should think about...", "Let's stay in touch", "It would be nice if..."
- Questions without answers: "Should we review the contract?"
- Informational statements: "Aditya is on Slack if you need help", "The team uses Jira"
- Expectations of third parties: "They'll probably get back to us", "That feature is coming in Q3"
- Casual past mentions with no new info: "We set up a bi-weekly cadence" (already established routine)
- Conditional wishes: "If you're looking forward to having a session, we could set one up"
- Status observations: "Numbers should be whatever they're doing behind the scenes"
</not_commitments>

**IMPORTANT — Status updates ARE valid extractions:**
If someone reports progress on a previously committed task (e.g., "I sent the budget analysis
to finance yesterday, they're reviewing it now"), extract it as a STATUS UPDATE with
`is_status_update=true` and the appropriate `implied_status`. These help track completion
of earlier commitments.

### Step 2: Evaluate Each Signal Against the Commitment Framework

For each commitment signal you identified, verify it passes these criteria:

1. **Decision**: What was agreed? (Not just a discussion topic — a closed decision)
2. **Action**: What is the specific deliverable? (Not "follow up" but "send the pricing deck")
3. **Owner**: Can you identify ONE person responsible? (Not a team, not vague)
4. **Timeline**: Is there any time indication? (Even approximate — note if missing)
5. **Definition of Done**: What proves completion? (If you can't define it, it may not be actionable)

**If a signal fails criteria 1-3, DO NOT extract it.** If it fails only 4-5, extract it but assign lower specificity scores.

## EXTRACTION LIMITS

Extract the **3 to 5 most significant** action items per transcript. If the transcript genuinely contains more high-quality commitments, you may extract up to 8, but NEVER more than 8.

**Prioritize:**
- Deal-advancing commitments over administrative tasks
- Explicit commitments over weak intentions
- Items with clear owners over unattributed tasks
- Time-bound items over open-ended ones

**Consolidate sub-tasks:** If multiple items are sub-steps of the same deliverable, consolidate into a single action item. For example:
- "Send the pricing deck, the SOC2 report, and the security whitepaper" = ONE item ("Send evaluation materials package")
- "Start planning the renewal" + "Communicate the renewal plan to the customer" = ONE item ("Plan and communicate renewal strategy")

## ITEM TYPES

Extract TWO types of items:

1. **NEW ACTION ITEMS**: Fresh commitments made during this conversation
2. **STATUS UPDATES**: References to previously committed items (e.g., "I sent that deck", "Done!")

## SCORING GUIDE

For each extracted item, assign scores on a 1-5 scale:

**Impact** (score_impact):
- 5: Deal-critical — blocks a purchase decision, legal requirement, revenue at stake
- 4: High-value — advances deal stage, satisfies key stakeholder request
- 3: Important — standard follow-up with clear business value
- 2: Nice-to-have — informational, relationship maintenance
- 1: Trivial — minor administrative task

**Urgency** (score_urgency):
- 5: Blocking/overdue — someone is waiting right now
- 4: Hard deadline within 48 hours
- 3: Expected within a week
- 2: Expected within a month, no hard deadline
- 1: No time pressure, whenever convenient

**Specificity** (score_specificity):
- 5: Crystal clear — anyone could execute ("Send Q1 pricing deck to sarah@acme.com by Friday 5pm")
- 4: Clear deliverable, minor details to fill in ("Schedule a demo with their tech team next week")
- 3: Understood in context but needs clarification for someone else
- 2: Vague deliverable ("Follow up with them")
- 1: Unclear what "done" looks like

**Effort** (score_effort):
- 5: Multi-day project (build a POC, draft a full proposal)
- 4: Half-day task (prepare a presentation, write a detailed analysis)
- 3: 1-2 hour task (draft an email, schedule a complex meeting)
- 2: 15-30 minute task (send an existing document, make a quick call)
- 1: 5 minutes or less (forward an email, send a calendar invite)

## Topic Guidelines

For each action item, identify the HIGH-LEVEL TOPIC it belongs to.

Topic naming rules:
- Use 3-5 words that capture the strategic initiative or project
- Be consistent: related action items in the SAME conversation should share the SAME topic name
- Use title case (e.g., "Q1 Sales Expansion" not "q1 sales expansion")

Good topic examples:
- "Hire 3 SDRs by March" → Topic: "Q1 Sales Team Expansion"
- "Review security audit findings" → Topic: "Annual Security Compliance"
- "Send updated pricing deck" → Topic: "Enterprise Deal Proposal"

Bad topic examples (do NOT use):
- Too specific: "Hire SDR Named John" (should be broader initiative)
- Too broad: "Work", "Tasks", "Sales" (not specific enough)
- Too vague: "Stuff To Do", "Follow Ups" (not meaningful)

## Speaker Attribution Rules

Transcripts often use diarization labels (A:, B:, C:, Speaker 1:, etc.) instead of
real names. Follow this hierarchy for the `owner` field:

1. **Named** — If you can identify the speaker by name from conversational context
   (introductions, how others address them, self-identification), use their real name.
   Example: Speaker A says "Hey Jackie" and Jackie responds → owner is "Jackie".

2. **Role-inferred** — If you cannot determine a name but can infer the speaker's role
   from context (who gives the demo, who asks about pricing, who manages the account),
   use a descriptive role. Examples: "the account executive", "the client engineer",
   "the project lead".

3. **Unconfirmed** — If neither name nor role can be determined, set owner to
   "unconfirmed". Do NOT use bare diarization labels (A, B, C, E, Speaker 1) as owner.

Set `owner_type` to reflect which tier was used: "named", "role_inferred", or "unconfirmed".

**Summary phrasing by owner type:**
- Named: "Sarah will send the proposal by Friday"
- Role-inferred: "The account executive will send the proposal by Friday"
- Unconfirmed: "Send the proposal by Friday" (use imperative voice — no fake attribution)

**User identification:**
If the recording user's name is provided in the context below, identify which speaker
corresponds to them using contextual clues (who opens the meeting, who introduces others,
who says "I'll follow up"). Set `is_user_owned` to true for action items owned by the
recording user. This does NOT filter items — extract everything, just tag the user's items.

## OUTPUT RULES

- Set has_action_items=false if no genuine commitments are found
- Use extraction_notes to explain if you filtered out borderline items
- The confidence field should reflect extraction certainty (did the person clearly commit?)
- Set commitment_strength to categorize the nature of the commitment
- Do NOT extract items with commitment_strength='observation' — those are not action items
- Do NOT extract vague intentions ("we should think about...")
- Do NOT extract questions without commitments
- Do NOT extract past actions that don't require follow-up"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Extract actionable commitments from the following transcript.

<context>
{additional_context}
</context>

<transcript>
{transcript_text}
</transcript>

<instructions>
1. Identify commitment signals in the transcript (first-person future tense, explicit promises, etc.)
2. Filter each signal against the commitment framework (Decision, Action, Owner, Timeline, Done)
3. Consolidate related sub-tasks into single items where appropriate
4. Score each surviving item on impact, urgency, specificity, and effort
5. Return the 3-5 most significant items (max 8 if truly warranted)
</instructions>"""


def build_extraction_prompt(
    transcript_text: str,
    meeting_title: str | None = None,
    participants: list[str] | None = None,
    user_name: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the extraction prompt messages for OpenAI.

    Args:
        transcript_text: The transcript text to extract from
        meeting_title: Optional meeting title for context
        participants: Optional list of participant names
        user_name: Optional name of the recording user (for is_user_owned tagging)

    Returns:
        List of message dicts for OpenAI chat completion
    """
    # Build additional context if available
    context_parts = []
    if meeting_title:
        context_parts.append(f'Meeting title: {meeting_title}')
    if participants:
        participant_lines = '\n'.join(f'  - {p}' for p in participants)
        context_parts.append(f"Meeting participants:\n{participant_lines}")
    if user_name:
        context_parts.append(f'Recording user: {user_name}')

    additional_context = '\n'.join(context_parts) if context_parts else ''

    user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
        transcript_text=transcript_text,
        additional_context=additional_context,
    )

    return [
        {'role': 'system', 'content': EXTRACTION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


# =============================================================================
# Deduplication Prompt (for matching phase)
# =============================================================================


# Valid merge recommendations
MergeRecommendationType = Literal['merge', 'update_status', 'create_new', 'link_related']


class DeduplicationDecision(BaseModel):
    """Decision on whether two action items are the same."""

    is_same_item: bool = Field(
        ...,
        description='True if the new extraction refers to the SAME real-world task as the '
        'existing action item. False if they are different tasks.',
    )
    is_status_update: bool = Field(
        default=False,
        description='True if the new extraction is a STATUS UPDATE about the existing item '
        '(e.g., marking it complete, providing progress). False otherwise.',
    )
    merge_recommendation: MergeRecommendationType = Field(
        ...,
        description='Action to take: "merge" (combine into existing), "update_status" (just update '
        'status of existing), "create_new" (create as separate item), "link_related" '
        '(create new but link as related).',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of why this decision was made.',
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description='Confidence in this decision (0.0 to 1.0).',
    )


DEDUPLICATION_SYSTEM_PROMPT = """You are an expert at determining whether action items refer to the same real-world task.

Given an EXISTING ACTION ITEM and a NEW EXTRACTION, determine if they refer to the SAME task.

They are the SAME ITEM if:
- They describe the same action to be done by the same person
- The new extraction is a status update about the existing item
- The new extraction adds details/context to the existing item
- They have the same deliverable, just described differently

They are DIFFERENT ITEMS if:
- They involve different people as the owner
- They are related but distinct tasks (e.g., "send proposal" vs "review proposal")
- They have fundamentally different deliverables
- They are for different recipients/purposes

Make a recommendation:
- **merge**: The new extraction should be combined with the existing item (same task, new info)
- **update_status**: The new extraction is just a status update (e.g., "I did that")
- **create_new**: These are genuinely different tasks, create a new action item
- **link_related**: Different tasks but clearly related, create new but link them"""

DEDUPLICATION_USER_PROMPT_TEMPLATE = """Compare these action items and determine if they refer to the same task.

<existing_action_item>
Text: {existing_text}
Owner: {existing_owner}
Summary: {existing_summary}
Status: {existing_status}
Created: {existing_created}
</existing_action_item>

<new_extraction>
Text: {new_text}
Owner: {new_owner}
Summary: {new_summary}
Is Status Update: {new_is_status_update}
Context: {new_context}
</new_extraction>

<similarity_score>{similarity_score:.3f}</similarity_score>

Are these the same action item?"""


def build_deduplication_prompt(
    existing_text: str,
    existing_owner: str,
    existing_summary: str,
    existing_status: str,
    existing_created: str,
    new_text: str,
    new_owner: str,
    new_summary: str,
    new_is_status_update: bool,
    new_context: str,
    similarity_score: float,
) -> list[dict[str, str]]:
    """
    Build the deduplication prompt messages for OpenAI.

    Args:
        existing_*: Fields from the existing ActionItem
        new_*: Fields from the newly extracted item
        similarity_score: Cosine similarity between embeddings

    Returns:
        List of message dicts for OpenAI chat completion
    """
    user_prompt = DEDUPLICATION_USER_PROMPT_TEMPLATE.format(
        existing_text=existing_text,
        existing_owner=existing_owner,
        existing_summary=existing_summary,
        existing_status=existing_status,
        existing_created=existing_created,
        new_text=new_text,
        new_owner=new_owner,
        new_summary=new_summary,
        new_is_status_update=str(new_is_status_update),
        new_context=new_context,
        similarity_score=similarity_score,
    )

    return [
        {'role': 'system', 'content': DEDUPLICATION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
