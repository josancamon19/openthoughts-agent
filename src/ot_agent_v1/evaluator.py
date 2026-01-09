"""
Debug evaluator using Claude Agent SDK.

Launches Claude Code to investigate task completions via SSH.
"""

import asyncio
import threading

from claude_agent_sdk import query, ClaudeAgentOptions


HARBOR_DEBUG_PROMPT = """
## Harbor Task Structure

You are debugging an agent task run. Here's the container structure:

### Key Paths to Check
- **Working directory** - Check with `pwd` after SSH, typically `/workspace` or /app or the root of the project
- `/tests/test.sh` - The test script that verifies the solution (READ THIS FIRST)
- `/logs/verifier/reward.txt` - Contains `1` (pass) or `0` (fail)
- `/logs/agent/` - Agent logs and trajectory files
- `/solution/` - Reference solution (if available, contains `solve.sh`)

### Verification Flow
1. Agent modifies files in the working directory
2. Tests run via `bash /tests/test.sh`
3. Test script writes reward to `/logs/verifier/reward.txt`

### Your Task
SSH in and quickly determine why it passed or failed:
1. Read `/tests/test.sh` to understand what's being verified
2. Check the working directory for what the agent created/modified
3. Compare against test expectations
4. Give a brief explanation (2-3 sentences) of what went wrong or right
"""


async def run_debug_agent(
    ssh_command: str,
    task_name: str,
    test_passed: bool | None,
    test_output: str | None,
    instruction: str | None = None,
):
    """
    Run Claude Code via Agent SDK to debug a task completion.

    Args:
        ssh_command: SSH command to access the container
        task_name: Name of the task being debugged
        test_passed: Whether tests passed (True/False/None)
        test_output: Raw test output (if available)
        instruction: Original task instruction

    Yields:
        Messages from Claude Agent SDK stream
    """
    status_str = (
        "✅ PASSED"
        if test_passed
        else "❌ FAILED"
        if test_passed is False
        else "⚠️ UNKNOWN"
    )

    prompt_parts = [
        f"# Debug Task: {task_name}",
        f"**Result:** {status_str}",
        HARBOR_DEBUG_PROMPT,
        f"\n## SSH Command\n```bash\n{ssh_command}\n```",
    ]

    if instruction:
        prompt_parts.append(f"\n## Task Given to Agent\n```\n{instruction[:2000]}\n```")

    if test_output:
        prompt_parts.append(f"\n## Test Output\n```\n{test_output[:2000]}\n```")

    prompt_parts.append(
        "\nSSH in, read the test script, check the agent's work, and explain briefly why it passed/failed."
    )

    full_prompt = "\n".join(prompt_parts)

    async for message in query(
        prompt=full_prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="bypassPermissions",
        ),
    ):
        yield message


def stream_debug_agent(
    ssh_command: str,
    task_name: str,
    test_passed: bool | None,
    test_output: str | None,
    instruction: str | None = None,
):
    """
    Synchronous wrapper that yields debug agent messages.

    This runs the async debug agent in a thread and yields results.
    """
    import queue

    result_queue = queue.Queue()
    done_event = threading.Event()

    async def collect_messages():
        try:
            async for msg in run_debug_agent(
                ssh_command=ssh_command,
                task_name=task_name,
                test_passed=test_passed,
                test_output=test_output,
                instruction=instruction,
            ):
                result_queue.put(msg)
        except Exception as e:
            result_queue.put({"error": str(e)})
        finally:
            done_event.set()

    def run_in_thread():
        asyncio.run(collect_messages())

    thread = threading.Thread(target=run_in_thread)
    thread.start()

    while not done_event.is_set() or not result_queue.empty():
        try:
            msg = result_queue.get(timeout=0.1)
            yield msg
        except queue.Empty:
            continue

    thread.join()
