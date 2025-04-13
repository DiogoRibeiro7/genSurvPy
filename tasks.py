from invoke.tasks import task

@task
def test(c):
    c.run("poetry run pytest --cov=gen_surv --cov-report=term --cov-report=xml")

@task
def docs(c):
    c.run("poetry run sphinx-build docs/source docs/build")

@task
def stubs(c):
    c.run("poetry run stubgen -p gen_surv -o stubs")

@task
def build(c):
    c.run("poetry build")

@task
def publish(c):
    c.run("poetry publish --build")

@task
def clean(c):
    c.run("rm -rf dist build docs/build .pytest_cache .mypy_cache coverage.xml .coverage stubs")
    
@task
def git_push(c):
    """
    Stage all changes, prompt for a commit message, create a signed commit, and push.
    """
    import getpass

    c.run("git add .")

    try:
        # Prompt for a commit message
        message = input("Enter commit message: ").strip()
        if not message:
            print("Aborting: empty commit message.")
            return

        c.run(f'git commit -S -m "{message}"')
        c.run("git push")
    except KeyboardInterrupt:
        print("\nAborted by user.")

