---
allowed-tools: Bash(git add:*, git commit:*, git status:*, git diff:*, git log:*)
description: Create a git commit with proper staging and message format
---

## Context

- Current git status: !`git status`
- Recent commits for reference: !`git log --oneline -5`
- Staged and unstaged changes: !`git diff HEAD`

## Your task

Create a single git commit following the project's conventions:

1. **Stage all changes** with `git add -A` (as per CLAUDE.md requirement)
2. **Create a commit** with the format: `type(scope): description (TICKET-ID)`
   - Types: feat, fix, refactor, docs, test, chore, etc.
   - Use appropriate scope based on the changes
   - Include ticket ID if available (e.g., ORQ-123, TOPS-000, INN-858)
3. **Include standard footer**:

```plain
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

Use a HEREDOC format for the commit message to ensure proper formatting:

```bash
git commit -m "$(cat <<'EOF'
feat(core): implement new feature (ORQ-123)

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**Important**: 

- Never update git config
- Only commit when changes exist
- Follow the exact format from CLAUDE.md
- Always stage ALL changes with `git add -A`