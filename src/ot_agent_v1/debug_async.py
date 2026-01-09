#!/usr/bin/env python3
"""
Debug script to simulate Streamlit's async behavior and test event loop issues.

This script simulates what happens in Streamlit when you run agents multiple times.
The key issue is that asyncio.run() creates and closes event loops, but the Daytona
SDK's aiohttp client caches references to the old (closed) event loop.

Run this without Streamlit to debug the "Event loop is closed" error.

Usage:
    cd src/ot_agent_v1
    python debug_async.py
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# MOCK TASK DATA - simulate what would come from the RL dataset
# ============================================================================

MOCK_TASK_INSTRUCTION = """
Create a simple shell script called hello.sh that prints "Hello, World!" to stdout.
"""


def create_mock_task_dir() -> Path:
    """Create a minimal mock task directory for testing."""
    tmpdir = Path(tempfile.mkdtemp(prefix="debug_task_"))

    # Create instruction.md
    (tmpdir / "instruction.md").write_text(MOCK_TASK_INSTRUCTION)

    # Create minimal task.toml
    (tmpdir / "task.toml").write_text("""
[environment]
base_image = "python:3.11-slim"
seed_commands = ["apt-get update && apt-get install -y bash"]
build_timeout_sec = 300
agent_timeout_sec = 300
""")

    # Create tests directory
    tests_dir = tmpdir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("""#!/bin/bash
if [ -f /workspace/hello.sh ]; then
    output=$(bash /workspace/hello.sh)
    if [ "$output" = "Hello, World!" ]; then
        echo "1" > /logs/verifier/reward.txt
        exit 0
    fi
fi
echo "0" > /logs/verifier/reward.txt
exit 1
""")

    # Create solution directory (optional)
    solution_dir = tmpdir / "solution"
    solution_dir.mkdir()
    (solution_dir / "hello.sh").write_text('#!/bin/bash\necho "Hello, World!"')

    return tmpdir


# ============================================================================
# FIX 1: Use a persistent event loop in a dedicated thread
# ============================================================================

import threading
import queue
from typing import Any, Coroutine


class AsyncRunner:
    """
    A persistent async runner that maintains a single event loop in a background thread.
    This avoids the "Event loop is closed" error by keeping the loop alive between calls.
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    def _run_loop(self):
        """Run the event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def start(self):
        """Start the background event loop thread."""
        if self._thread is None or not self._thread.is_alive():
            self._started.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._started.wait()
            print(f"[AsyncRunner] Started background event loop")

    def run(self, coro: Coroutine) -> Any:
        """Run a coroutine in the persistent event loop."""
        self.start()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self):
        """Stop the event loop and thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop = None
            self._thread = None
            print(f"[AsyncRunner] Stopped background event loop")


# Global async runner instance
_async_runner: AsyncRunner | None = None


def get_async_runner() -> AsyncRunner:
    """Get or create the global async runner."""
    global _async_runner
    if _async_runner is None:
        _async_runner = AsyncRunner()
    return _async_runner


def run_async_persistent(coro: Coroutine) -> Any:
    """
    Run an async coroutine using a persistent event loop.
    This is the FIX for the "Event loop is closed" error.
    """
    return get_async_runner().run(coro)


# ============================================================================
# FIX 2: Reset Daytona SDK state between runs
# ============================================================================


def reset_daytona_state():
    """
    Reset any cached state in the Daytona SDK.
    This includes clearing module caches and forcing fresh connections.
    """
    import gc

    # Force garbage collection first
    gc.collect()

    # Clear cached modules that might hold stale connections
    modules_to_clear = [
        k
        for k in list(sys.modules.keys())
        if any(
            prefix in k
            for prefix in [
                "daytona",
                "harbor.environments",
                "aiohttp",
                "daytona_api_client_async",
            ]
        )
    ]

    for mod in modules_to_clear:
        del sys.modules[mod]

    print(f"[reset_daytona_state] Cleared {len(modules_to_clear)} cached modules")

    # Force garbage collection again
    gc.collect()


# ============================================================================
# TEST SCENARIOS
# ============================================================================


async def simple_async_test():
    """A simple async function to test event loop behavior."""
    print("[test] Running simple async test...")
    await asyncio.sleep(0.1)
    print("[test] Simple async test completed")
    return "OK"


async def daytona_connection_test():
    """Test Daytona SDK connection (requires DAYTONA_API_KEY)."""
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("[test] DAYTONA_API_KEY not set, skipping Daytona test")
        return "SKIPPED"

    print("[test] Testing Daytona connection...")

    # Import Daytona SDK
    from daytona import AsyncDaytona

    # Create client and test connection
    daytona = AsyncDaytona(
        api_key=api_key,
        server_url="https://app.daytona.io/api",
    )

    # List sandboxes (simple API call to test connection)
    print("[test] Listing sandboxes...")
    sandboxes = await daytona.list()
    print(f"[test] Found {len(sandboxes)} sandboxes")

    return "OK"


async def full_agent_test():
    """Test the full run_agent flow (requires DAYTONA_API_KEY + ANTHROPIC_API_KEY)."""
    api_key = os.environ.get("DAYTONA_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("[test] DAYTONA_API_KEY not set, skipping agent test")
        return "SKIPPED"
    if not anthropic_key:
        print("[test] ANTHROPIC_API_KEY not set, skipping agent test")
        return "SKIPPED"

    # Import here to avoid circular imports
    from env import run_agent

    # Create mock task
    task_dir = create_mock_task_dir()
    print(f"[test] Created mock task at: {task_dir}")

    try:
        result = await run_agent(
            task_dir=task_dir,
            instruction=MOCK_TASK_INSTRUCTION,
            daytona_api_key=api_key,
            agent_name="claude-code",
            model_name="claude-sonnet-4-5-20250929",  # Use Sonnet for faster/cheaper test
            status_log=[],
        )

        print(f"[test] Agent run completed:")
        print(f"  - sandbox_id: {result['sandbox_id']}")
        print(f"  - test_passed: {result['test_passed']}")

        return result
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(task_dir, ignore_errors=True)


# ============================================================================
# MAIN TEST HARNESS
# ============================================================================


def test_with_asyncio_run(test_fn, num_runs=2):
    """
    Test using asyncio.run() - this will likely fail on run 2+.
    This simulates what happens in the current Streamlit code.
    """
    print("\n" + "=" * 60)
    print(f"TEST: {test_fn.__name__} with asyncio.run() ({num_runs} runs)")
    print("=" * 60)

    for i in range(num_runs):
        print(f"\n--- Run {i + 1} ---")
        try:
            # This is what the current code does - it will fail on subsequent runs
            # because the Daytona SDK caches event loop references

            # Clear modules (same as current run_async)
            modules_to_clear = [
                k
                for k in sys.modules.keys()
                if k.startswith(("daytona", "harbor.environments"))
            ]
            for mod in modules_to_clear:
                sys.modules.pop(mod, None)

            result = asyncio.run(test_fn())
            print(f"Result: {result}")
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                print(f"❌ FAILED: {e}")
                print("   This is the bug we're trying to fix!")
            else:
                raise


def test_with_persistent_loop(test_fn, num_runs=2):
    """
    Test using the persistent event loop runner - this should work.
    This is the FIX for the event loop issue.
    """
    print("\n" + "=" * 60)
    print(f"TEST: {test_fn.__name__} with persistent loop ({num_runs} runs)")
    print("=" * 60)

    runner = AsyncRunner()

    try:
        for i in range(num_runs):
            print(f"\n--- Run {i + 1} ---")
            try:
                result = runner.run(test_fn())
                print(f"Result: {result}")
            except Exception as e:
                print(f"❌ FAILED: {e}")
                import traceback

                traceback.print_exc()
    finally:
        runner.stop()


def test_with_reset_state(test_fn, num_runs=2):
    """
    Test using asyncio.run() but with full state reset between runs.
    """
    print("\n" + "=" * 60)
    print(f"TEST: {test_fn.__name__} with state reset ({num_runs} runs)")
    print("=" * 60)

    for i in range(num_runs):
        print(f"\n--- Run {i + 1} ---")
        try:
            # Reset all Daytona/aiohttp state
            reset_daytona_state()

            # Run in fresh thread with fresh event loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, test_fn())
                result = future.result()

            print(f"Result: {result}")
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                print(f"❌ FAILED: {e}")
            else:
                raise


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main entry point for debug script."""
    import argparse

    parser = argparse.ArgumentParser(description="Debug async event loop issues")
    parser.add_argument(
        "--test",
        choices=["simple", "daytona", "agent", "all"],
        default="daytona",
        help="Which test to run",
    )
    parser.add_argument(
        "--method",
        choices=["asyncio_run", "persistent", "reset", "all"],
        default="all",
        help="Which async method to test",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs to test",
    )

    args = parser.parse_args()

    # Select test function
    if args.test == "simple":
        test_fn = simple_async_test
    elif args.test == "daytona":
        test_fn = daytona_connection_test
    elif args.test == "agent":
        test_fn = full_agent_test
    else:
        # Run all tests
        for fn in [simple_async_test, daytona_connection_test]:
            for method in ["asyncio_run", "persistent", "reset"]:
                if method == "asyncio_run":
                    test_with_asyncio_run(fn, args.runs)
                elif method == "persistent":
                    test_with_persistent_loop(fn, args.runs)
                else:
                    test_with_reset_state(fn, args.runs)
        return

    # Run selected test with selected method
    if args.method == "asyncio_run":
        test_with_asyncio_run(test_fn, args.runs)
    elif args.method == "persistent":
        test_with_persistent_loop(test_fn, args.runs)
    elif args.method == "reset":
        test_with_reset_state(test_fn, args.runs)
    else:
        # Run all methods
        test_with_asyncio_run(test_fn, args.runs)
        test_with_persistent_loop(test_fn, args.runs)
        test_with_reset_state(test_fn, args.runs)

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    print("""
SUMMARY:
- asyncio_run: Will fail on run 2+ due to cached event loop refs in Daytona SDK
- persistent: Uses a single background event loop (RECOMMENDED FIX)
- reset: Aggressively clears all cached modules (may work but slower)

To fix in your Streamlit app, update env.py's run_async() to use the 
persistent AsyncRunner pattern from this script.
""")


if __name__ == "__main__":
    main()
