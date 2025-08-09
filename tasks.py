import shlex

from invoke import Context, task


@task
def test(c: Context) -> None:
    """Run the test suite with coverage reporting.

    This task executes ``pytest`` and generates a coverage report.

    Args:
        c: Invoke context used to run shell commands.

    Raises:
        TypeError: If ``c`` is not an Invoke :class:`Context`.
    """
    # Ensure we were passed a valid Context object.
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r} instead")

    # Build the command string. You can adjust '--cov=gen_surv' if you
    # need to cover a different package or add extra pytest flags.
    command = (
        "poetry run pytest " "--cov=gen_surv " "--cov-report=term " "--cov-report=xml"
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
        exit_code = (
            result.exited
            if result is not None and hasattr(result, "exited")
            else "Unknown"
        )
        print(f"Exit code: {exit_code}")
        stderr_output = (
            result.stderr if result is not None and hasattr(result, "stderr") else None
        )
        if stderr_output:
            print("Error output:")
            print(stderr_output)


@task
def docs(c: Context) -> None:
    """Build the Sphinx documentation.

    Args:
        c: Invoke context used to run shell commands.

    Raises:
        TypeError: If ``c`` is not an Invoke :class:`Context`.
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
        exit_code = (
            result.exited
            if result is not None and hasattr(result, "exited")
            else "Unknown"
        )
        print(f"Exit code: {exit_code}")
        stderr_output = (
            result.stderr if result is not None and hasattr(result, "stderr") else None
        )
        if stderr_output:
            print("Error output:")
            print(stderr_output)


@task
def stubs(c: Context) -> None:
    """Generate type stubs for the ``gen_surv`` package.

    Args:
        c: Invoke context used to run shell commands.

    Raises:
        TypeError: If ``c`` is not an Invoke :class:`Context`.
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
        exit_code = (
            result.exited
            if result is not None and hasattr(result, "exited")
            else "Unknown"
        )
        print(f"Exit code: {exit_code}")
        stderr_output = (
            result.stderr if result is not None and hasattr(result, "stderr") else None
        )
        if stderr_output:
            print("Error output:")
            print(stderr_output)


@task
def build(c: Context) -> None:
    """Build distribution artifacts using Poetry.

    Args:
        c: Invoke context used to run shell commands.

    Raises:
        TypeError: If ``c`` is not an Invoke :class:`Context`.
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
        print(
            "✔️  Build completed successfully. Artifacts are in the 'dist/' directory."
        )
    else:
        print("❌  Build failed.")
        exit_code = (
            result.exited
            if result is not None and hasattr(result, "exited")
            else "Unknown"
        )
        print(f"Exit code: {exit_code}")
        stderr_output = (
            result.stderr if result is not None and hasattr(result, "stderr") else None
        )
        if stderr_output:
            print("Error output:")
            print(stderr_output)


@task
def publish(c: Context) -> None:
    """Build and upload the package to PyPI.

    Args:
        c: Invoke context used to run shell commands.
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
    """Remove build artifacts and caches.

    Args:
        c: Invoke context used to run shell commands.

    Raises:
        TypeError: If ``c`` is not an Invoke :class:`Context`.
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
    """Commit and push all staged changes.

    Args:
        c: Invoke context used to run shell commands.

    Raises:
        TypeError: If ``c`` is not an Invoke :class:`Context`.
    """
    # Verify the argument is a valid Invoke Context.
    if not isinstance(c, Context):
        raise TypeError(f"Expected Invoke Context, got {type(c).__name__!r}")

    # Stage all changes.
    result_add = c.run("git add .", warn=True, pty=False)
    if result_add is None or not getattr(result_add, "ok", False):
        print("❌ Failed to stage changes (git add).")
        exit_code = (
            result_add.exited
            if result_add is not None and hasattr(result_add, "exited")
            else "Unknown"
        )
        print(f"Exit code: {exit_code}")
        stderr_output = (
            result_add.stderr
            if result_add is not None and hasattr(result_add, "stderr")
            else None
        )
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
            exit_code = (
                getattr(result_push, "exited", "Unknown")
                if result_push is not None
                else "Unknown"
            )
            print(f"Exit code: {exit_code}")
            stderr_output = (
                getattr(result_push, "stderr", None)
                if result_push is not None
                else None
            )
            if stderr_output:
                print("Error output:")
                print(stderr_output)
    except KeyboardInterrupt:
        print("\nAborted by user.")
