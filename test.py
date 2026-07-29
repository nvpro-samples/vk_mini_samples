import argparse
import shlex
import subprocess
import logging
import sys
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_ARGS = ["--headless"]

# Appended to every sample (including the overrides below) so failures print
# detailed Vulkan context information to help diagnose errors.
VERBOSE_ARGS = ["--verbose", "1"]

SAMPLES = [
    "barycentric_wireframe",
    "compute_multi_threaded",
    "compute_only",
    "descriptor_heap",
    "gltf_raytrace",
    "image_ktx",
    "image_viewer",
    "line_stipple",
    "memory_budget",
    "mesh_shaders",
    "mesh_task_shaders",
    "mm_opacity",
    "msaa",
    "offscreen",
    "ray_query",
    "ray_query_position_fetch",
    "ray_trace",
    "ray_trace_clusters",
    "ray_trace_motion_blur",
    "ray_tracing_position_fetch",
    "realtime_analysis",
    "rectangle",
    "ser_pathtrace",
    "shader_object",
    "shader_printf",
    "simple_polygons",
    "solid_color",
    "texture_3d",
    "tiny_shader_toy",
]

SAMPLE_ARGS_OVERRIDE = {
    "offscreen": [],
    "gltf_raytrace": ["--headless", "--frames", "100"],
    "ray_query": ["--headless", "--frames", "100"],
}


def get_sample_args(name):
    return SAMPLE_ARGS_OVERRIDE.get(name, DEFAULT_ARGS) + VERBOSE_ARGS


class ReturnCode(Enum):
    SUCCESS = 0
    TEST_ERROR = 1
    ENVIRONMENT_ERROR = 2


@dataclass
class TestResult:
    name: str
    status: ReturnCode
    time: str


def _append_captured_output(log_file, header, stdout=None, stderr=None):
    """Append captured process output to the sample log file (best-effort)."""
    try:
        with log_file.open("a", encoding="utf-8", errors="replace") as f:
            f.write(f"\n\n===== {header} =====\n")
            if stdout and stdout.strip():
                f.write("--- stdout ---\n")
                f.write(stdout.rstrip())
                f.write("\n")
            if stderr and stderr.strip():
                f.write("--- stderr ---\n")
                f.write(stderr.rstrip())
                f.write("\n")
    except OSError as e:
        logger.error(f"Could not append output to {log_file}: {e}")


def run_command(commands, timeout=None, log_file=None):
    """Run a command and return the process return code."""
    logger.info(f"Running command: {' '.join(commands)}")
    try:
        result = subprocess.run(
            commands, text=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout}s: {' '.join(commands)}")
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(
            "utf-8", errors="replace"
        )
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode(
            "utf-8", errors="replace"
        )
        if stdout and stdout.strip():
            logger.error(f"stdout (partial):\n{stdout.rstrip()}")
        if stderr and stderr.strip():
            logger.error(f"stderr (partial):\n{stderr.rstrip()}")
        if log_file is not None:
            _append_captured_output(
                log_file, f"TIMEOUT after {timeout}s", stdout, stderr
            )
        return -1

    if result.returncode != 0:
        logger.error(
            f"Command failed (exit code {result.returncode}): {' '.join(commands)}"
        )
        if result.stdout and result.stdout.strip():
            logger.error(f"stdout:\n{result.stdout.rstrip()}")
        if result.stderr and result.stderr.strip():
            logger.error(f"stderr:\n{result.stderr.rstrip()}")
        if log_file is not None:
            _append_captured_output(
                log_file,
                f"FAILED exit code {result.returncode}",
                result.stdout,
                result.stderr,
            )
    else:
        logger.debug(f"Command output:\n{result.stdout}\n{result.stderr}")
    return result.returncode


def extract_testing_time(log_file):
    """Extract testing time from the last line of the log file.
    Example log line: '[00:000:01.993]  -> 84.683 ms'
    Returns the value after '->' or 'N/A' if not found."""
    if log_file.exists():
        content = log_file.read_text()
        lines = content.splitlines()
        if lines:
            last_line = lines[-1].strip()
            if "->" in last_line:
                try:
                    return last_line.split("->")[1].strip()
                except IndexError:
                    pass
    return "N/A"


def build_effective_args(
    predefined_args,
    *,
    no_headless=False,
    override_args=None,
    extra_args=None,
    frames=None,
):
    """Build the final argument list for an executable from predefined + overrides."""
    if override_args is not None:
        args = list(override_args)
    else:
        args = list(predefined_args)

    if no_headless:
        args = [a for a in args if a != "--headless"]

    if frames is not None:
        if "--frames" in args:
            idx = args.index("--frames")
            if idx + 1 < len(args):
                args[idx + 1] = str(frames)
            else:
                args.append(str(frames))
        else:
            args.extend(["--frames", str(frames)])

    if extra_args:
        args.extend(extra_args)

    return args


def find_executable(test_dir, executable):
    """Resolve executable path, accounting for .exe on Windows."""
    path = test_dir / executable
    if path.exists():
        return path
    exe_path = path.with_suffix(".exe")
    if exe_path.exists():
        return exe_path
    return None


def test_executable(test_dir, executable, args, timeout=None):
    """Test a single executable and return its TestResult."""
    executable_path = find_executable(test_dir, executable)
    if executable_path is None:
        logger.error(f"Executable not found: {test_dir / executable}[.exe]")
        return TestResult(executable, ReturnCode.ENVIRONMENT_ERROR, "N/A")

    try:
        log_file = test_dir / f"log_{executable}.txt"
        return_code = run_command(
            [str(executable_path)] + args, timeout=timeout, log_file=log_file
        )
        testing_time = extract_testing_time(log_file)
        status = ReturnCode.SUCCESS if return_code == 0 else ReturnCode.TEST_ERROR
        return TestResult(executable, status, testing_time)
    except Exception as e:
        logger.error(f"Error testing {executable}: {e}")
        return TestResult(executable, ReturnCode.TEST_ERROR, "N/A")


def resolve_test_list(
    sample_filter=None,
    *,
    no_headless=False,
    override_args=None,
    extra_args=None,
    frames=None,
):
    """Return list of (name, effective_args) tuples after applying all overrides."""
    names = SAMPLES
    if sample_filter:
        names = [n for n in names if sample_filter in n]
        if not names:
            logger.error(f"No sample matching '{sample_filter}'. Available samples:")
            for name in SAMPLES:
                logger.error(f"  {name}")
            return None

    return [
        (
            name,
            build_effective_args(
                get_sample_args(name),
                no_headless=no_headless,
                override_args=override_args,
                extra_args=extra_args,
                frames=frames,
            ),
        )
        for name in names
    ]


def run_tests(test_dir, tests, timeout=None):
    """Run the given list of (name, args) tests."""
    if not test_dir.exists():
        logger.error(f"Test directory '{test_dir}' does not exist.")
        return [], ReturnCode.ENVIRONMENT_ERROR

    results = []
    overall_status = ReturnCode.SUCCESS

    for executable, args in tests:
        logger.info(f"Testing: {executable}")
        result = test_executable(test_dir, executable, args, timeout=timeout)
        results.append(result)
        if result.status != ReturnCode.SUCCESS:
            overall_status = ReturnCode.TEST_ERROR

    return results, overall_status


def print_report(results):
    """Print a formatted report of test results."""
    passed = sum(1 for r in results if r.status == ReturnCode.SUCCESS)
    failed = len(results) - passed

    logger.info("")
    logger.info("Test Results:")
    logger.info("-" * 80)
    logger.info("{:<35} | {:<12} | {:<8}".format("Executable", "Status", "Time"))
    logger.info("-" * 80)

    for r in results:
        status_str = "SUCCESS" if r.status == ReturnCode.SUCCESS else "FAILED"
        logger.info("{:<35} | {:<12} | {:<8}".format(r.name, status_str, r.time))
    logger.info("-" * 80)
    logger.info(f"{passed}/{len(results)} passed, {failed} failed")

    failures = [r for r in results if r.status != ReturnCode.SUCCESS]
    if failures:
        logger.info("")
        logger.info("Re-run failing tests individually:")
        for r in failures:
            logger.info(f"  python test.py --sample {r.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Test vk_mini_samples executables.",
        epilog="Examples:\n"
        "  python test.py                            Run all tests\n"
        "  python test.py --sample ray_trace         Run only ray_trace\n"
        "  python test.py --no-headless              Local visual run (no --headless)\n"
        "  python test.py --frames 200               Override frame count\n"
        "  python test.py --extra-args '--width 800' Append extra arguments\n"
        "  python test.py --override-args '--headless --frames 50'\n"
        "                                            Replace all predefined args\n"
        "  python test.py --list                     Show samples + effective args\n"
        "  python test.py --dry-run                  Preview without executing\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--test", action="store_true", default=True, help=argparse.SUPPRESS)

    filt = parser.add_argument_group("filtering")
    filt.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Run only samples whose name contains this substring",
    )

    arg_group = parser.add_argument_group("argument overrides")
    arg_group.add_argument(
        "--no-headless",
        action="store_true",
        help="Strip --headless from all sample arguments (local visual run)",
    )
    arg_group.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Set or override --frames value for all samples",
    )
    arg_group.add_argument(
        "--extra-args",
        type=str,
        default=None,
        help="Extra arguments appended to each sample (quoted string, e.g. '--width 800')",
    )
    arg_group.add_argument(
        "--override-args",
        type=str,
        default=None,
        help="Completely replace predefined arguments (quoted string)",
    )

    run_group = parser.add_argument_group("execution")
    run_group.add_argument(
        "--test-dir",
        type=Path,
        default=Path("_install"),
        help="Directory containing built executables (default: _install)",
    )
    run_group.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds per executable (default: 60)",
    )
    run_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show full command output (debug logging)",
    )
    run_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running anything",
    )
    run_group.add_argument(
        "--list",
        action="store_true",
        help="List all sample names with effective arguments and exit",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    extra = shlex.split(args.extra_args) if args.extra_args else None
    override = shlex.split(args.override_args) if args.override_args else None

    tests = resolve_test_list(
        args.sample,
        no_headless=args.no_headless,
        override_args=override,
        extra_args=extra,
        frames=args.frames,
    )
    if tests is None:
        return ReturnCode.ENVIRONMENT_ERROR

    if args.list or args.dry_run:
        label = "Would run:" if args.dry_run else "Available samples:"
        print(f"\n{label}")
        print("-" * 80)
        for name, sample_args in tests:
            print(f"  {name:<35} {' '.join(sample_args)}")
        print("-" * 80)
        print(f"  {len(tests)} sample(s)")
        return ReturnCode.SUCCESS

    results, status = run_tests(args.test_dir, tests, args.timeout)
    if results:
        print_report(results)
    return status


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(result.value)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(ReturnCode.TEST_ERROR.value)
