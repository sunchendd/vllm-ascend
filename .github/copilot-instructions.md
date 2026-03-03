# Copilot 工作区指令

## 可用 Skills

### vllm-dev-workflow
vLLM 开发验证工作流，用于 Ascend NPU 环境下的 vLLM 推理服务开发、测试和验证。

**详细文档**: [`skills/vllm-dev-workflow/SKILL.md`](../skills/vllm-dev-workflow/SKILL.md)

**触发场景**:
- vLLM 开发验证 / 投机解码测试
- 自适应性能对比 / 基线对比测试
- 启动服务验证

---


## Project overview

A collection of Claude Code agent skills for development workflows, distributed as a plugin marketplace. Contains 5 skills organized into 2 plugins:

- **github-workflow** plugin: `git-commit`, `github-pr-creation`, `github-pr-merge`, `github-pr-review`
- **skill-authoring** plugin: `creating-skills`

Skills are model-invoked (Claude activates them based on user intent, not slash commands).

## Architecture

```
.claude-plugin/
  marketplace.json        # Plugin registry: defines plugins, skill paths, metadata
skills/
  <skill-name>/
    SKILL.md              # Main skill file (YAML frontmatter + markdown body)
    references/           # Optional deep-dive docs loaded on demand
```

### Key file: `marketplace.json`

Defines the plugin structure. Each plugin has a `skills` array pointing to skill directories. When adding or removing skills, update both the plugin's `skills` array and the corresponding directory.

### Skill anatomy

Every skill requires a `SKILL.md` with:
1. **YAML frontmatter** (`name` + `description`) - the description is critical for discovery, it determines when Claude activates the skill
2. **Markdown body** - workflow instructions, kept under 500 lines

Reference files in `references/` provide extended examples and documentation that Claude loads only when needed (progressive disclosure).

## Conventions

- **Commits**: Conventional Commits format - `type(scope): subject` (see `skills/git-commit/SKILL.md`)
- **Naming**: lowercase, hyphens between words, no spaces (e.g., `github-pr-review`)
- **Merge strategy**: always merge commits (`--merge`), never squash/rebase
- **Changelog**: follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
- **Versioning**: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## Writing skills

When creating or editing skills, follow the patterns in `skills/creating-skills/SKILL.md`:

- Description formula: `<What it does>. Use when <trigger phrases>. <Key capabilities>.`
- SKILL.md body under 500 lines; move detailed content to `references/`
- Only create helper scripts when they add real value (complex processing, JSON transformation), not for single-command wrappers
- Mark critical constraints with bold **ALWAYS**/**NEVER** in "Important Rules" sections
- Include trigger phrases in descriptions so Claude activates the skill on the right user intents
