# Real Transcript Testing

This directory contains real transcripts for end-to-end pipeline testing.

## How to Use

1. Add your transcripts to `transcripts.json` following the schema below
2. Run the test runner: `python examples/run_transcript_tests.py`
3. Review results in the console output and `results/` directory

## Transcript Schema

Edit `transcripts.json` with your transcripts:

```json
{
  "tenant_id": "YOUR-TENANT-UUID",
  "account_id": "acct_your_account_id",
  "account_name": "Optional Account Name",
  "transcripts": [
    {
      "sequence": 1,
      "meeting_title": "Initial Discovery Call",
      "timestamp": "2024-01-15T10:00:00Z",
      "participants": ["John Smith", "Sarah Jones", "Client: Mike"],
      "text": "Your transcript text here...",
      "expected_action_items": 3,
      "notes": "Optional notes about what to expect"
    },
    {
      "sequence": 2,
      "meeting_title": "Follow-up Call",
      "timestamp": "2024-01-22T14:00:00Z",
      "participants": ["John Smith", "Sarah Jones"],
      "text": "Follow-up transcript with status updates...",
      "expected_action_items": 2,
      "expected_updates": 1,
      "notes": "Should match item from call 1"
    }
  ]
}
```

## Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `tenant_id` | Yes | UUID for your tenant (will be used for all transcripts) |
| `account_id` | Yes | Account identifier (same across all transcripts for matching) |
| `account_name` | No | Human-readable account name |
| `transcripts[].sequence` | Yes | Processing order (1, 2, 3...) |
| `transcripts[].meeting_title` | No | Title for context |
| `transcripts[].timestamp` | No | ISO8601 timestamp |
| `transcripts[].participants` | No | List of participant names |
| `transcripts[].text` | Yes | The transcript content |
| `transcripts[].expected_action_items` | No | Expected count for validation |
| `transcripts[].expected_updates` | No | Expected update count for validation |
| `transcripts[].notes` | No | Your notes about expectations |

## Example Workflow

**Transcript 1 (Initial Call):**
```
Sarah: I'll send the proposal to Acme by Friday.
John: And I'll schedule the demo for next week.
```
Expected: 2 new action items created

**Transcript 2 (Follow-up Call):**
```
Sarah: I sent the proposal yesterday, they're reviewing it.
John: Great. I also scheduled the demo for Thursday.
```
Expected: 2 items matched and updated to completed/in_progress

**Transcript 3 (New Action Items):**
```
Sarah: We need to prepare a custom ROI analysis.
John: I'll handle that before the next call.
```
Expected: 1-2 new items, previous items unchanged
