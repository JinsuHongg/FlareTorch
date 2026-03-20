# Contributing to FlareTorch

Thank you for your interest in contributing to FlareTorch! This document provides guidelines and instructions for contributing to the project.

## Development Setup

FlareTorch uses [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jhong90/FlareTorch.git
    cd FlareTorch
    ```

2.  **Install dependencies:**
    ```bash
    uv sync
    ```

3.  **Run tests:**
    ```bash
    uv run pytest
    ```

## Commit Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries

Example: `feat: add flare detection module`

## Pull Request Guidelines

1.  **Create a new branch** for your changes:
    ```bash
    git checkout -b feature/your-feature-name
    ```
2.  **Ensure tests pass** before submitting a PR.
3.  **Write clear commit messages** following the conventions above.
4.  **Update documentation** if your changes introduce new features or change existing behavior.
5.  **Submit the PR** with a clear description of the changes and the problem they solve.

## Authors

- Jinsu Hong
- Berkay Aydin
