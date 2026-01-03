"""
Daytona environment management for OpenThoughts Agent.
"""

import asyncio
import io
import tarfile
import tempfile
from pathlib import Path

from daytona import AsyncDaytona, CreateSandboxFromImageParams, DaytonaConfig, Image

DAYTONA_API_URL = "https://app.daytona.io/api"  


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


async def create_daytona_sandbox(api_key: str, dockerfile_path: Path) -> dict:
    """Create a Daytona sandbox from a Dockerfile."""
    config = DaytonaConfig(
        api_key=api_key,
        api_url=DAYTONA_API_URL,
    )

    async with AsyncDaytona(config=config) as daytona:
        image = Image.from_dockerfile(dockerfile_path)
        params = CreateSandboxFromImageParams(
            image=image,
            auto_delete_interval=1800,  # 30 min auto-delete
        )
        sandbox = await daytona.create(params=params, timeout=300)
        ssh_access = await sandbox.create_ssh_access()
        return {
            "sandbox_id": sandbox.id,
            "ssh_command": f"ssh {ssh_access.token}@ssh.app.daytona.io",
        }


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
