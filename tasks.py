from invoke.tasks import task
from invoke import Context, task
from typing import Any
import shlex




@task
def test(c: Context) -> None:
    """
    Run pytest via Poetry with coverage reporting for the 'gen_surv' package.

    This task will:
      1. Execute 'pytest' through Poetry.
      2. Generate a terminal coverage report.
      3. Write an XML coverage report to 'coverage.xml'.

    :param c: Invoke context used to run shell commands.
    :raises TypeError: If 'c' is not an Invoke Context.
    """
    # Ensure we were passed a valid Context object.
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r} instead")

    # Build the command string. You can adjust '--cov=gen_surv' if you
    # need to cover a different package or add extra pytest flags.
    command = (
        "poetry run pytest "
        "--cov=gen_surv "
        "--cov-report=term "
        "--cov-report=xml"
    )

    # Run pytest. 
    # - warn=True: capture non-zero exit codes without aborting Invoke.
    # - pty=False: pytest doesn’t require an interactive TTY here.
    result = c.run(command, warn=True, pty=False)

    # Check the exit code and report accordingly.
    if result is not None and getattr(result, "ok", False):
        print("✔️  All tests passed.")
    else:
        print("❌  Some tests failed.")
        exit_code = result.exited if result is not None and hasattr(result, "exited") else "Unknown"
        print(f"Exit code: {exit_code}")
        stderr_output = result.stderr if result is not None and hasattr(result, "stderr") else None
        if stderr_output:
            print("Error output:")
            print(stderr_output)


@task
def checkversion(c: Context) -> None:
    """Validate that ``pyproject.toml`` matches the latest git tag.

    This task runs the ``scripts/check_version_match.py`` helper using Poetry
    and reports whether the version numbers are aligned.

    Args:
        c: Invoke context used to run shell commands.

    Returns:
        None
    """
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r}")

    # Execute the version check script with Poetry.
    cmd = "poetry run python scripts/check_version_match.py"
    result = c.run(cmd, warn=True, pty=False)

    # Report based on the exit code from the script.
    if result.ok:
        print("✔️  pyproject version matches the latest git tag.")
    else:
        print("❌  Version mismatch detected.")
        print(result.stderr)

@task
def docs(c: Context) -> None:
    """
    Build Sphinx documentation for the project using Poetry.

    This task will:
      1. Run 'sphinx-build' via Poetry.
      2. Read source files from 'docs/source'.
      3. Output HTML (or other format) into 'docs/build'.

    :param c: Invoke context, used to run shell commands.
    :type c: Context
    :raises TypeError: If 'c' is not an Invoke Context.
    """
    # Verify we have a proper Invoke Context.
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r}")

    # Construct the Sphinx build command. Adjust paths if needed.
    command = "poetry run sphinx-build docs/source docs/build"

    # Execute sphinx-build.
    # - warn=True: capture non-zero exits without immediately aborting Invoke.
    # - pty=False: sphinx-build does not require interactive input.
    result = c.run(command, warn=True, pty=False)

    # Report on the result of the documentation build.
    if result is not None and getattr(result, "ok", False):
        print("✔️  Documentation built successfully.")
    else:
        print("❌  Documentation build failed.")
        exit_code = result.exited if result is not None and hasattr(result, "exited") else "Unknown"
        print(f"Exit code: {exit_code}")
        stderr_output = result.stderr if result is not None and hasattr(result, "stderr") else None
        if stderr_output:
            print("Error output:")
            print(stderr_output)


@task
def stubs(c: Context) -> None:
    """
    Generate type stubs for the 'gen_surv' package using stubgen and Poetry.

    This task will:
      1. Run 'stubgen' via Poetry to analyze 'gen_surv'.
      2. Output the generated stubs into the 'stubs' directory.

    :param c: Invoke context used to run shell commands.
    :raises TypeError: If 'c' is not an Invoke Context.
    """
    # Verify that 'c' is the correct Invoke Context.
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r}")

    # Build the stubgen command. Adjust '-p gen_surv' or output path if needed.
    command = "poetry run stubgen -p gen_surv -o stubs"

    # Execute stubgen.
    # - warn=True: capture non-zero exit codes without aborting Invoke.
    # - pty=False: stubgen does not require interactive input.
    result = c.run(command, warn=True, pty=False)

    # Report on the outcome of stub generation.
    if result is not None and getattr(result, "ok", False):
        print("✔️  Type stubs generated successfully in 'stubs/'.")
    else:
        print("❌  Stub generation failed.")
        exit_code = result.exited if result is not None and hasattr(result, "exited") else "Unknown"
        print(f"Exit code: {exit_code}")
        stderr_output = result.stderr if result is not None and hasattr(result, "stderr") else None
        if stderr_output:
            print("Error output:")
            print(stderr_output)


@task
def build(c: Context) -> None:
    """
    Build the project distributions using Poetry.

    This task will:
      1. Run 'poetry build' to create source and wheel packages.
      2. Place the built artifacts in the 'dist/' directory.

    :param c: Invoke context used to run shell commands.
    :raises TypeError: If 'c' is not an Invoke Context.
    """
    # Verify that we received a valid Invoke Context.
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r}")

    # Construct the build command. Adjust if you need custom build options.
    command = "poetry build"

    # Execute the build.
    # - warn=True: capture non-zero exit codes without aborting Invoke.
    # - pty=False: no interactive input is required for building.
    result = c.run(command, warn=True, pty=False)

    # Report the result of the build process.
    if result is not None and getattr(result, "ok", False):
        print("✔️  Build completed successfully. Artifacts are in the 'dist/' directory.")
    else:
        print("❌  Build failed.")
        exit_code = result.exited if result is not None and hasattr(result, "exited") else "Unknown"
        print(f"Exit code: {exit_code}")
        stderr_output = result.stderr if result is not None and hasattr(result, "stderr") else None
        if stderr_output:
            print("Error output:")
            print(stderr_output)

@task
def publish(c: Context) -> None:
    """
    Build and publish the package to PyPI using Poetry.

    This task will:
      1. Build the distribution via 'poetry publish --build'.
      2. Attach to a pseudo-TTY so you can enter credentials or confirm prompts.
      3. Not abort immediately if an error occurs; instead, it will print diagnostics.

    :param c: Invoke context, used to run shell commands.
    :type c: Context
    """
    # Run the poetry publish command.
    # - warn=True: do not abort on non-zero exit, so we can inspect and report.
    # - pty=True: allocate a pseudo-TTY for interactive prompts (username/password, etc.).
    result = c.run(
        "poetry publish --build",
        warn=True,
        pty=True,
    )

    # If the exit code is zero, the publish succeeded.
    if result.ok:
        print("✔️ Package published successfully.")
        return

    # Otherwise, print out details to help debug.
    print("❌ Poetry publish failed.")
    print(f"Exit code: {result.exited}")
    if result.stderr:
        print("Error output:")
        print(result.stderr)
    else:
        print("No stderr output captured.")

@task
def clean(c: Context) -> None:
    """
    Remove build artifacts, caches, and generated files.

    This task will:
      1. Delete the 'dist' and 'build' directories.
      2. Remove generated documentation in 'docs/build'.
      3. Clear pytest and mypy caches.
      4. Delete coverage reports and stub files.

    :param c: Invoke context used to run shell commands.
    :raises TypeError: If 'c' is not an Invoke Context.
    """
    # Verify the argument is an Invoke Context.
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r}")

    # List of paths and files to remove. Adjust if you add new artifacts.
    targets = [
        "dist",
        "build",
        "docs/build",
        ".pytest_cache",
        ".mypy_cache",
        "coverage.xml",
        ".coverage",
        "stubs",
    ]

    # Join targets into a single rm command.
    # Using '-rf' to force removal without prompts.
    command = f"rm -rf {' '.join(targets)}"

    # Execute the cleanup command.
    # - warn=True: capture non-zero exits without aborting Invoke.
    # - pty=False: no interactive input is required.
    result = c.run(command, warn=True, pty=False)

    # Report the outcome of the cleanup.
    if result.ok:
        print("✔️  Cleaned all build artifacts and caches.")
    else:
        print("❌  Cleanup failed for some targets.")
        print(f"Exit code: {result.exited}")
        if result.stderr:
            print("Error output:")
            print(result.stderr)
    
@task
def gitpush(c: Context) -> None:
    """
    Stage all changes, prompt for a commit message, create a signed commit, and push to the remote repository.

    This task will:
      1. Verify that 'c' is an Invoke Context.
      2. Run 'git add .' to stage all unstaged changes.
      3. Prompt the user for a commit message; abort if empty.
      4. Sanitize the message, then run 'git commit -S -m <message>'.
      5. Run 'git push' to publish commits.

    :param c: Invoke Context used to run shell commands.
    :raises TypeError: If 'c' is not an Invoke Context.
    """
    # Verify the argument is a valid Invoke Context.
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r}")

    # Stage all changes.
    result_add = c.run("git add .", warn=True, pty=False)
    if result_add is None or not getattr(result_add, "ok", False):
        print("❌ Failed to stage changes (git add).")
        exit_code = result_add.exited if result_add is not None and hasattr(result_add, "exited") else "Unknown"
        print(f"Exit code: {exit_code}")
        stderr_output = result_add.stderr if result_add is not None and hasattr(result_add, "stderr") else None
        if stderr_output:
            print("Error output:")
            print(stderr_output)
        return

    try:
        # Prompt for a commit message.
        message = input("Enter commit message: ").strip()
        if not message:
            print("Aborting: empty commit message.")
            return

        # Sanitize the message to prevent shell injection.
        sanitized_message = shlex.quote(message)

        # Create a signed commit. Use a pseudo-TTY so GPG passphrase can be entered if needed.
        result_commit = c.run(
            f"git commit -S -m {sanitized_message}",
            warn=True,
            pty=True,
        )
        if result_commit is None or not getattr(result_commit, "ok", False):
            print("❌ Commit failed.")
            exit_code = getattr(result_commit, "exited", "Unknown")
            print(f"Exit code: {exit_code}")
            stderr_output = getattr(result_commit, "stderr", None)
            if stderr_output:
                print("Error output:")
                print(stderr_output)
            return

        # Push to the remote repository.
        result_push = c.run("git push", warn=True, pty=False)
        if result_push is not None and getattr(result_push, "ok", False):
            print("✔️  Changes pushed successfully.")
        else:
            print("❌ Push failed.")
            exit_code = getattr(result_push, "exited", "Unknown") if result_push is not None else "Unknown"
            print(f"Exit code: {exit_code}")
            stderr_output = getattr(result_push, "stderr", None) if result_push is not None else None
            if stderr_output:
                print("Error output:")
                print(stderr_output)
    except KeyboardInterrupt:
        print("\nAborted by user.")
