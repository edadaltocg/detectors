import subprocess


def get_last_version() -> str:
    """Return the version number of the last release."""
    string = (
        subprocess.run(
            ["gh", "release", "list", "--limit", "1"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode("utf8")
        .strip()
    )
    return string.split()[-2]


def bump_patch_number(version_number: str) -> str:
    """Return a copy of `version_number` with the patch number incremented."""
    major, minor, patch = version_number.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


def create_new_patch_release():
    """Create a new patch release on GitHub."""
    try:
        last_version_number = get_last_version()
    except subprocess.CalledProcessError as err:
        if err.stderr.decode("utf8").startswith("HTTP 404:"):
            # The project doesn't have any releases yet.
            new_version_number = "0.0.1"
        else:
            raise
    else:
        new_version_number = bump_patch_number(last_version_number)

    subprocess.run(
        ["gh", "release", "create", "--generate-notes", new_version_number],
        check=True,
    )


if __name__ == "__main__":
    create_new_patch_release()
