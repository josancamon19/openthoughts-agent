"""
Daytona environment management for OpenThoughts Agent using Harbor framework.

## Supported Agents

1. **Claude Code** (claude_code):
   - Type: Installed agent (runs inside container)
   - Installation: npm via nvm
   - Log location: `/logs/agent/sessions/projects/-app/*.jsonl`
   - Command structure: 2 commands (setup dirs, then run agent)
   - API key: ANTHROPIC_API_KEY
   - Model format: "claude-opus-4-5-20251101"

2. **Terminus2** (terminus_2):
   - Type: External agent (runs outside container, interfaces via environment)
   - Log location: `/logs/agent/trajectory.json`
   - API key: ANTHROPIC_API_KEY
   - Model format: "anthropic/claude-opus-4-5-20251101" (LiteLLM format)

## Testing Agents
   - SSH into container to verify installation: `which claude`
   - Check install logs: `cat /installed-agent/install.sh`
   - Check agent output: `cat /logs/agent/*.txt`
"""

import asyncio
import io
import json
import os
import tarfile
import tempfile
from enum import Enum
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

DAYTONA_API_URL = "https://app.daytona.io/api"


class AgentType(Enum):
    """Supported agent types for the harness."""

    CLAUDE_CODE = "claude_code"
    TERMINUS_2 = "terminus_2"


# Agent configuration including display names, API key requirements, and model formats
AGENT_CONFIG = {
    AgentType.CLAUDE_CODE: {
        "display_name": "Claude Code",
        "agent_class": "installed",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key_description": "Anthropic API Key",
        "default_model": "claude-opus-4-5-20251101",
        "log_file": "sessions/projects/-app/*.jsonl",
    },
    AgentType.TERMINUS_2: {
        "display_name": "Terminus2",
        "agent_class": "external",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key_description": "Anthropic API Key",
        "default_model": "anthropic/claude-opus-4-5-20251101",
        "log_file": "trajectory.json",
    },
}


def extract_task_to_tempdir(task_binary: bytes) -> Path | None:
    """Extract task tarball to a temp directory and return the path."""
    try:
        tmpdir = tempfile.mkdtemp(prefix="rl_task_")
        tar_io = io.BytesIO(task_binary)
        with tarfile.open(fileobj=tar_io, mode="r:gz") as tar:
            tar.extractall(tmpdir, filter="data")
        return Path(tmpdir)
    except Exception:
        return None


async def create_harbor_daytona_env(api_key: str, task_dir: Path) -> dict:
    """
    Create a Daytona sandbox using Harbor's task infrastructure.
    This properly handles all task files including seeds, tests, and solution.
    """
    # Set Daytona credentials in environment
    os.environ["DAYTONA_API_KEY"] = api_key
    os.environ["DAYTONA_API_URL"] = DAYTONA_API_URL

    from harbor.environments.daytona import DaytonaEnvironment
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

    # Load task using Harbor's Task class
    task = Task(task_dir)

    # Create temporary trial paths
    temp_trial_dir = tempfile.mkdtemp(prefix="harbor_trial_")
    trial_paths = TrialPaths(trial_dir=Path(temp_trial_dir))

    # Create DaytonaEnvironment
    env = DaytonaEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=task.name,
        session_id=str(uuid4()),
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
    )

    # Start the environment (builds container with all files)
    await env.start(force_build=True)

    # Upload solution and tests
    if task.paths.solution_dir.exists():
        await env.upload_dir(
            task.paths.solution_dir, str(EnvironmentPaths.solution_dir)
        )
    if task.paths.tests_dir.exists():
        await env.upload_dir(task.paths.tests_dir, str(EnvironmentPaths.tests_dir))

    # Get SSH access
    ssh_access = await env._sandbox.create_ssh_access()

    return {
        "sandbox_id": env._sandbox.id,
        "ssh_command": f"ssh {ssh_access.token}@ssh.app.daytona.io",
        "task_name": task.name,
        "environment": env,
    }


def _get_agent_class(agent_type: AgentType):
    """Import and return the appropriate agent class based on agent type."""
    if agent_type == AgentType.CLAUDE_CODE:
        from harbor.agents.installed.claude_code import ClaudeCode

        return ClaudeCode
    elif agent_type == AgentType.TERMINUS_2:
        from harbor.agents.terminus_2.terminus_2 import Terminus2

        return Terminus2
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def run_agent(
    task_dir: Path,
    instruction: str,
    daytona_api_key: str,
    agent_type: AgentType = AgentType.CLAUDE_CODE,
    model_name: str | None = None,
    status_log: list = None,
) -> dict:
    """
    Run an agent on a task in a Daytona environment.

    Args:
        task_dir: Path to the extracted task directory
        instruction: The task instruction to give the agent
        daytona_api_key: Daytona API key
        agent_type: Which agent to use (default: CLAUDE_CODE)
        model_name: Model to use (default: agent-specific default)
        status_log: Optional list to collect status messages

    Returns:
        Dict with sandbox_id, trajectory, and agent context
    """

    def status(msg):
        if status_log is not None:
            status_log.append(msg)
        print(f"[*] {msg}")

    # Get agent configuration
    agent_config = AGENT_CONFIG[agent_type]
    agent_display_name = agent_config["display_name"]

    # Use agent-specific default model if not provided
    if model_name is None:
        model_name = agent_config["default_model"]

    # Set Daytona credentials
    os.environ["DAYTONA_API_KEY"] = daytona_api_key
    os.environ["DAYTONA_API_URL"] = DAYTONA_API_URL

    # Verify API key based on agent requirements
    api_key_env = agent_config["api_key_env"]
    if not os.environ.get(api_key_env):
        raise ValueError(f"{api_key_env} not set in environment")

    # Import Harbor components
    from harbor.environments.daytona import DaytonaEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

    # Get the appropriate agent class
    AgentClass = _get_agent_class(agent_type)

    # Load task
    task = Task(task_dir)

    # Create logs directory for agent
    logs_dir = Path(tempfile.mkdtemp(prefix=f"{agent_type.value}_logs_"))

    # Create trial paths
    trial_dir = Path(tempfile.mkdtemp(prefix="harbor_trial_"))
    trial_paths = TrialPaths(trial_dir=trial_dir)

    # Create DaytonaEnvironment
    session_id = str(uuid4())
    env = DaytonaEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=task.name,
        session_id=session_id,
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
    )

    status(f"Starting environment for task: {task.name}")
    await env.start(force_build=True)

    # Get SSH access early so user can connect while agent runs
    ssh_access = await env._sandbox.create_ssh_access()
    ssh_command = f"ssh {ssh_access.token}@ssh.app.daytona.io"
    status(f"🔗 SSH ready: {ssh_command}")

    # Upload solution and tests
    if task.paths.solution_dir.exists():
        status("Uploading task files...")
        await env.upload_dir(
            task.paths.solution_dir, str(EnvironmentPaths.solution_dir)
        )

    if task.paths.tests_dir.exists():
        await env.upload_dir(task.paths.tests_dir, str(EnvironmentPaths.tests_dir))

    # Create the agent with appropriate parameters
    agent = AgentClass(logs_dir=logs_dir, model_name=model_name)

    # Create agent context
    context = AgentContext()

    # Determine agent class type (installed vs external)
    is_external_agent = agent_config.get("agent_class") == "external"

    # Setup agent
    if is_external_agent:
        status(f"Setting up {agent_display_name} agent...")
    else:
        status(f"Installing {agent_display_name} agent...")
    await agent.setup(env)

    # Run the agent
    status(f"Running {agent_display_name}...")

    # Track exec_commands for later use (only for installed agents)
    exec_commands = []

    if is_external_agent:
        # External agents (like Terminus2) use the run() method directly
        status(f"{agent_display_name} is working on the task...")
        await agent.run(instruction, env, context)
    else:
        # Installed agents use create_run_agent_commands and manual execution
        from harbor.utils.templating import render_prompt_template

        rendered_instruction = instruction
        if agent._prompt_template_path:
            rendered_instruction = render_prompt_template(
                agent._prompt_template_path, instruction
            )

        exec_commands = agent.create_run_agent_commands(rendered_instruction)
        num_commands = len(exec_commands)

        for i, exec_input in enumerate(exec_commands):
            command_dir = logs_dir / f"command-{i}"
            command_dir.mkdir(parents=True, exist_ok=True)
            (command_dir / "command.txt").write_text(exec_input.command)

            # Only show meaningful progress, not raw commands
            if num_commands == 1:
                status(f"{agent_display_name} is working on the task...")
            elif i == 0:
                status("Setting up agent environment...")
            else:
                status(f"{agent_display_name} is working on the task...")

            result = await env.exec(
                command=exec_input.command,
                cwd=exec_input.cwd,
                env=exec_input.env,
                timeout_sec=exec_input.timeout_sec,
            )

            (command_dir / "return-code.txt").write_text(str(result.return_code))
            if result.stdout:
                (command_dir / "stdout.txt").write_text(result.stdout)
            if result.stderr:
                (command_dir / "stderr.txt").write_text(result.stderr)

            # Report command completion with return code
            if result.return_code != 0:
                status(f"⚠️ Command {i} failed with exit code {result.return_code}")
                # Show stderr/stdout for debugging
                if result.stderr:
                    print(f"[DEBUG] Command {i} stderr:\n{result.stderr[:1000]}")
                if result.stdout:
                    print(
                        f"[DEBUG] Command {i} stdout (first 1000 chars):\n{result.stdout[:1000]}"
                    )

    # Collect agent trajectory
    status("Collecting agent trajectory...")
    container_agent_dir = str(EnvironmentPaths.agent_dir)  # /logs/agent

    if is_external_agent:
        # External agents (like Terminus2) write trajectory directly to logs_dir
        # The trajectory.json should already be in logs_dir
        trajectory_path = logs_dir / "trajectory.json"
        if trajectory_path.exists():
            print(f"[DEBUG] Found trajectory at {trajectory_path}")
        else:
            print(f"[DEBUG] No trajectory.json found in {logs_dir}")
    else:
        # For installed agents, download logs from container
        # First, check what files exist in the agent logs directory
        ls_result = await env.exec(
            command=f"ls -la {container_agent_dir} 2>/dev/null || echo 'Directory not found'"
        )
        if ls_result.stdout:
            print(f"[DEBUG] Container agent dir contents:\n{ls_result.stdout}")

        download_success = False
        try:
            await env.download_dir(
                source_dir=container_agent_dir,
                target_dir=str(logs_dir),
            )
            # Check if we got the expected log files based on agent type
            if agent_type == AgentType.CLAUDE_CODE:
                sessions_dir = logs_dir / "sessions"
                if sessions_dir.exists():
                    import subprocess

                    find_result = subprocess.run(
                        ["find", str(sessions_dir), "-type", "f", "-name", "*.jsonl"],
                        capture_output=True,
                        text=True,
                    )
                    if find_result.stdout.strip():
                        download_success = True
        except Exception as e:
            print(f"[DEBUG] Download dir failed: {e}")

        # Fallback: Create expected log structure from stdout if download failed
        if not download_success:
            # Find the last command's stdout (where agent output typically is)
            stdout_file = None
            for i in range(len(exec_commands) - 1, -1, -1):
                candidate = logs_dir / f"command-{i}" / "stdout.txt"
                if candidate.exists():
                    stdout_file = candidate
                    break

            if stdout_file and stdout_file.exists():
                import shutil

                if agent_type == AgentType.CLAUDE_CODE:
                    # Create the expected sessions directory structure for Claude Code
                    sessions_dir = logs_dir / "sessions" / "projects" / "-app"
                    sessions_dir.mkdir(parents=True, exist_ok=True)
                    session_jsonl = sessions_dir / "session.jsonl"
                    shutil.copy(stdout_file, session_jsonl)

    # Parse trajectory from logs
    try:
        agent.populate_context_post_run(context)
    except Exception as e:
        print(f"[DEBUG] populate_context_post_run failed: {e}")
    status("Agent run complete!")

    # Run tests to verify the solution
    status("Running verification tests...")
    test_passed = None
    test_output = None
    try:
        # Create verifier log directory
        await env.exec(command="mkdir -p /logs/verifier")

        # Run the test script
        test_result = await env.exec(
            command="bash /tests/test.sh 2>&1",
            timeout_sec=60,
        )
        test_output = test_result.stdout or ""
        if test_result.stderr:
            test_output += "\n" + test_result.stderr

        # Read the reward file
        reward_result = await env.exec(
            command="cat /logs/verifier/reward.txt 2>/dev/null || echo 0"
        )
        reward_str = (reward_result.stdout or "0").strip()
        test_passed = reward_str == "1"

        if test_passed:
            status("✅ Tests PASSED!")
        else:
            status(f"❌ Tests FAILED (reward={reward_str})")
    except Exception as e:
        status(f"⚠️ Test verification error: {e}")
        test_passed = None

    # Load trajectory if available
    trajectory = None
    trajectory_path = logs_dir / "trajectory.json"
    if trajectory_path.exists():
        with open(trajectory_path) as f:
            trajectory = json.load(f)

    # Read raw agent output for debugging
    raw_output = None
    if is_external_agent:
        # For external agents, raw output is in the trajectory
        if trajectory:
            raw_output = json.dumps(trajectory, indent=2)
    else:
        # For installed agents, read from command stdout
        for i in range(len(exec_commands) - 1, -1, -1):
            agent_output_file = logs_dir / f"command-{i}" / "stdout.txt"
            if agent_output_file.exists():
                raw_output = agent_output_file.read_text()
                break

    return {
        "sandbox_id": env._sandbox.id,
        "ssh_command": ssh_command,
        "task_name": task.name,
        "agent_type": agent_type.value,
        "agent_display_name": agent_display_name,
        "model_name": model_name,
        "logs_dir": str(logs_dir),
        "trajectory": trajectory,
        "raw_output": raw_output,
        "test_passed": test_passed,
        "test_output": test_output,
        "context": {
            "cost_usd": context.cost_usd,
            "n_input_tokens": context.n_input_tokens,
            "n_output_tokens": context.n_output_tokens,
            "n_cache_tokens": context.n_cache_tokens,
        },
        "environment": env,
    }


def run_async(coro):
    """Run async function in sync context - safe for Streamlit.

    Always runs in a fresh thread to avoid event loop caching issues
    with the Daytona SDK between Streamlit reruns.
    """
    import concurrent.futures
    import sys

    # Clear cached modules that might hold stale event loop references
    modules_to_clear = [
        k
        for k in sys.modules.keys()
        if k.startswith(("daytona", "harbor.environments"))
    ]
    for mod in modules_to_clear:
        sys.modules.pop(mod, None)

    # Always run in a new thread with a fresh event loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()
