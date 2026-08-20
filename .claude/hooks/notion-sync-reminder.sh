#!/usr/bin/env bash
# PostToolUse hook (Edit|Write): reminds Claude to mirror IMPROVEMENT_NOTES.md
# changes to the linked Notion page in the same session.
set -euo pipefail

NOTION_PAGE_ID="3c165974-eff0-81ba-9787-cb4928c01c3c"
NOTION_PAGE_URL="https://app.notion.com/p/3c165974eff081ba9787cb4928c01c3c"

file_path="$(jq -r '.tool_input.file_path // empty')"

if [[ "$file_path" == *IMPROVEMENT_NOTES.md ]]; then
  context="IMPROVEMENT_NOTES.md가 방금 수정되었습니다. 방금 바뀐 내용을 Notion 페이지(제목 \"⚾ KBO 투구 제구 예측 — 개선 노트\", URL ${NOTION_PAGE_URL}, ID ${NOTION_PAGE_ID})에도 mcp__claude_ai_Notion__notion-update-page 도구로 반영하세요. 전체를 다시 쓰지 말고 content_updates(부분 교체)나 insert_content(끝에 추가)로 방금 바뀐 섹션만 동기화하면 됩니다."
  jq -n --arg ctx "$context" '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $ctx}}'
fi
