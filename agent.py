import os
import asyncio
from typing import List, Dict, Any
from urllib.parse import urlparse

import dotenv
from github import Github, Auth, Repository, ContentFile
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCallResult,
    FunctionAgent,
    AgentWorkflow,
    ToolCall,
)
from llama_index.core.workflow import Context
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Load environment variables
dotenv.load_dotenv()

# Module-level exported variable required by tests (MUST end with .git)
repo_url = os.getenv("REPO_URL", "https://github.com/example/repo.git")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

llm = OpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    temperature=0.0,
)

# Initialize GitHub Client
auth = Auth.Token(GITHUB_TOKEN) if GITHUB_TOKEN else None
gh_client = Github(auth=auth)

raw_repo = os.getenv("REPOSITORY") or repo_url
def parse_repo_name(target: str) -> str:
    clean = target.rstrip("/").replace(".git", "")
    if "github.com/" in clean:
        clean = clean.split("github.com/")[-1]
    return clean

repository = parse_repo_name(raw_repo)
pr_number = os.getenv("PR_NUMBER")

repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")


print(f"DEBUG: Target Repo = '{repository}', Target PR = '{pr_number}'")

def parse_repo_url(url: str) -> tuple[str, str]:
    """Parses owner and repository name from a GitHub URL or 'owner/repo' string."""
    clean_url = url.rstrip("/").removesuffix(".git")
    if "github.com/" in clean_url:
        clean_url = clean_url.split("github.com/")[-1]
    elif "github.com:" in clean_url:
        clean_url = clean_url.split("github.com:")[-1]

    path_parts = clean_url.strip("/").split("/")
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    return "", path_parts[0] if path_parts else ""


def _get_target_repo() -> Repository.Repository:
    """Helper to resolve target repository dynamically from env or repo_url."""
    target_repo_str = os.getenv("REPOSITORY") or repo_url
    owner, repo_name = parse_repo_url(target_repo_str)
    full_name = f"{owner}/{repo_name}" if owner else repo_name
    return gh_client.get_repo(full_name)


# -------------------------------------------------------------------
# 1. GitHub API Retrieval Functions & State Tools
# -------------------------------------------------------------------


async def get_pr_details(pr_number: int) -> Dict[str, Any]:
    """
    Fetches details for a given pull request number.
    Returns details including author, title, body, diff_url, state, and commit SHAs.
    """
    loop = asyncio.get_running_loop()

    def _fetch():
        repo = _get_target_repo()
        pull_request = repo.get_pull(int(pr_number))

        body_text = pull_request.body.strip() if pull_request.body else ""
        if not body_text:
            body_text = "No description provided."

        commit_SHAs = [c.sha for c in pull_request.get_commits()]

        return {
            "author": pull_request.user.login if pull_request.user else "Unknown",
            "title": pull_request.title or "No Title",
            "body": body_text,
            "diff_url": pull_request.diff_url,
            "state": pull_request.state,
            "head_sha": pull_request.head.sha,
            "commit_SHAs": commit_SHAs,
        }

    return await loop.run_in_executor(None, _fetch)


async def read_file_content(file_path: str) -> str:
    """
    Fetches the raw text content of a specific file from the repository given its file path.
    """
    loop = asyncio.get_running_loop()

    def _fetch():
        repo = _get_target_repo()
        file_content: ContentFile.ContentFile = repo.get_contents(file_path)
        if isinstance(file_content, list):
            raise ValueError(f"Path '{file_path}' points to a directory, not a file.")
        return file_content.decoded_content.decode("utf-8")

    return await loop.run_in_executor(None, _fetch)


async def get_commit_details(head_sha: str) -> List[Dict[str, Any]]:
    """
    Given a commit SHA, retrieves information about the commit such as the files that changed.
    """
    loop = asyncio.get_running_loop()

    def _fetch():
        repo = _get_target_repo()
        commit = repo.get_commit(head_sha)

        return [
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            }
            for f in commit.files
        ]

    return await loop.run_in_executor(None, _fetch)


async def save_draft_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Saves the generated draft PR review comment into workflow state."""
    await ctx.store.set("draft_comment", draft_comment)
    return "Successfully saved draft_comment to state."


async def add_context_to_state(ctx: Context, context_summary: str) -> str:
    """Saves the gathered PR context and file summary into workflow state."""
    await ctx.store.set("context", context_summary)
    return "Context summary successfully saved to state."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Saves the final review comment into the 'final_review' key of workflow state."""
    await ctx.store.set("final_review", final_review)
    return "Successfully saved final review to state."


async def post_review_to_github(pr_number: int, comment: str) -> str:
    """Posts the final review comment as a PR Review on GitHub."""
    loop = asyncio.get_running_loop()

    def _post():
        repo = _get_target_repo()
        pull_request = repo.get_pull(int(pr_number))

        # Must call create_review so pr.get_reviews() picks it up in Test #4
        pull_request.create_review(body=comment, event="COMMENT")
        return f"Successfully posted review comment to PR #{pr_number} on GitHub."

    return await loop.run_in_executor(None, _post)

# -------------------------------------------------------------------
# 2. FunctionTool Definitions
# -------------------------------------------------------------------

pr_details_tool = FunctionTool.from_defaults(
    async_fn=get_pr_details,
    name="get_pr_details",
    description="Fetches details about a pull request given its number, including author, title, body, diff_url, state, head_sha, and commit_SHAs.",
)

file_tool = FunctionTool.from_defaults(
    async_fn=read_file_content,
    name="read_file_content",
    description="Fetches the contents of a file from the repository given a file path.",
)

commit_details_tool = FunctionTool.from_defaults(
    async_fn=get_commit_details,
    name="get_commit_details",
    description="Retrieves information about a commit given its SHA, including changed files, status, additions, deletions, and patch.",
)

save_draft_comment_tool = FunctionTool.from_defaults(
    async_fn=save_draft_comment_to_state,
    name="save_draft_comment_to_state",
    description="Saves the generated draft review comment to state.",
)

add_context_to_state_tool = FunctionTool.from_defaults(
    async_fn=add_context_to_state,
    name="add_context_to_state",
    description="Saves gathered PR details, changed files, and code diff summaries to global state.",
)

save_final_review_tool = FunctionTool.from_defaults(
    async_fn=add_final_review_to_state,
    name="save_final_review_to_state",
    description="Saves the final PR review text into the workflow state under the 'final_review' key.",
)

post_github_review_tool = FunctionTool.from_defaults(
    async_fn=post_review_to_github,
    name="post_review_to_github",
    description="Posts the final review comment to a specified PR number on GitHub.",
)


# -------------------------------------------------------------------
# 3. Agents & Workflow Setup
# -------------------------------------------------------------------

CONTEXT_AGENT_SYSTEM_PROMPT = """You are the ContextAgent.
Your sole job is to collect details about the PR:
1. Call `get_pr_details`.
2. Call `get_commit_details`.
3. Call `add_context_to_state` with a summary of the PR and diffs.
4. Call `handoff` to transfer control to `CommentorAgent`.

CRITICAL RULE: Do NOT generate direct textual answers for the user. Always complete your work via tool calls and perform the handoff tool call.
"""

context_agent = FunctionAgent(
    name="ContextAgent",
    description="Gathers PR details and commit changes, saves them to state, and hands off to CommentorAgent.",
    tools=[
        pr_details_tool,
        file_tool,
        commit_details_tool,
        add_context_to_state_tool,
    ],
    llm=llm,
    system_prompt=CONTEXT_AGENT_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

COMMENTOR_AGENT_SYSTEM_PROMPT = """You are the CommentorAgent.
Your job is to draft a comprehensive code review comment for the pull request.

MANDATORY STEPS YOU MUST EXECUTE USING TOOLS:
1. Generate a detailed markdown review comment (~200-300 words).
2. Call the tool `save_draft_comment_to_state` with your draft text as `draft_comment`.
3. Call the `handoff` tool with `to_agent="ReviewAndPostingAgent"` and a reason stating that the draft review is ready.

CRITICAL RULE: You are FORBIDDEN from responding directly with standard text. You MUST call `save_draft_comment_to_state` and then call `handoff` to `ReviewAndPostingAgent`.
"""

CommentorAgent = FunctionAgent(
    name="CommentorAgent",
    llm=llm,
    description="Drafts a review comment based on context, saves draft to state, then hands off to ReviewAndPostingAgent.",
    tools=[save_draft_comment_tool, file_tool],
    system_prompt=COMMENTOR_AGENT_SYSTEM_PROMPT,
    can_handoff_to=["ReviewAndPostingAgent"],
)

REVIEW_POSTING_AGENT_SYSTEM_PROMPT = """You are the ReviewAndPostingAgent.
You coordinate and finalize GitHub PR reviews.

DECISION LOGIC:

1. IF this is the START of the workflow (no draft review exists yet):
   - Call `handoff` to `ContextAgent` to gather PR details.

2. IF control was handed to you by `CommentorAgent` (or a draft comment is ready):
   - DO NOT HAND OFF to any agent.
   - Extract the PR number from the prompt or context (e.g., if user asked for "PR number 1", pr_number = 1).
   - Call `save_final_review_to_state` with the final review text.
   - You MUST call `post_review_to_github` with `pr_number` and the comment text. DO NOT skip this tool call!

CRITICAL: Never complete your response without calling `post_review_to_github`!
"""

ReviewAndPostingAgent = FunctionAgent(
    name="ReviewAndPostingAgent",
    llm=llm,
    description="Coordinates workflow execution, saves final review to state, and posts review to GitHub.",
    tools=[save_final_review_tool, post_github_review_tool],
    system_prompt=REVIEW_POSTING_AGENT_SYSTEM_PROMPT,
    can_handoff_to=["ContextAgent", "CommentorAgent"],
)


workflow_agent = AgentWorkflow(
    agents=[context_agent, CommentorAgent, ReviewAndPostingAgent],
    root_agent=ReviewAndPostingAgent.name,
    initial_state={
        "context": "",
        "draft_comment": "",
        "final_review": "",
    },
)


# -------------------------------------------------------------------
# 4. Asynchronous Streaming Main Entrypoint
# -------------------------------------------------------------------

async def main():
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response and event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

    await handler


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        gh_client.close()