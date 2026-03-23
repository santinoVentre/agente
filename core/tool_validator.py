"""ToolValidator — sandbox execution and security scanning for generated tools."""

from __future__ import annotations

import asyncio
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from agents.security_agent import SecurityAgent
from utils.logging import setup_logging

log = setup_logging("tool_validator")

security = SecurityAgent()


class ToolValidator:
    """Validates generated tool code: security scan + sandbox test execution."""

    def validate_code_security(self, code: str) -> tuple[bool, list[str]]:
        """Static security analysis of generated code."""
        return security.validate_generated_code(code)

    async def run_in_sandbox(
        self, code: str, test_code: str, timeout: int = 30
    ) -> tuple[bool, str]:
        """
        Execute generated tool code + tests in an isolated subprocess.
        Returns (success, output_or_error).
        """
        with tempfile.TemporaryDirectory(prefix="tool_sandbox_") as tmpdir:
            tool_file = Path(tmpdir) / "tool_module.py"
            test_file = Path(tmpdir) / "test_tool.py"

            tool_file.write_text(code, encoding="utf-8")

            # Wrap test code to import from the tool module
            full_test = textwrap.dedent(f"""\
                import sys
                sys.path.insert(0, {repr(tmpdir)})
                from tool_module import *
                import asyncio

                async def _run_tests():
                {textwrap.indent(test_code, '    ')}

                asyncio.run(_run_tests())
                print("ALL_TESTS_PASSED")
            """)
            test_file.write_text(full_test, encoding="utf-8")

            try:
                proc = await asyncio.create_subprocess_exec(
                    "python3", str(test_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmpdir,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode("utf-8", errors="replace")
                errors = stderr.decode("utf-8", errors="replace")

                if proc.returncode == 0 and "ALL_TESTS_PASSED" in output:
                    return True, output
                else:
                    return False, f"Exit code: {proc.returncode}\nStdout: {output}\nStderr: {errors}"

            except asyncio.TimeoutError:
                return False, f"Sandbox execution timed out after {timeout}s"
            except Exception as e:
                return False, f"Sandbox error: {e}"

    async def full_validation(
        self, code: str, test_code: str, max_retries: int = 0
    ) -> tuple[bool, str]:
        """
        Run both security scan and sandbox test.
        Returns (passed, details).
        """
        # 1. Security scan
        is_safe, issues = self.validate_code_security(code)
        if not is_safe:
            return False, f"Security issues found:\n" + "\n".join(f"  - {i}" for i in issues)

        # 2. Sandbox test
        passed, output = await self.run_in_sandbox(code, test_code)
        if not passed:
            return False, f"Sandbox test failed:\n{output}"

        return True, "All validations passed"


# Singleton
tool_validator = ToolValidator()
