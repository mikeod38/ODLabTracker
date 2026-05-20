Run the ODLabTracker dependency checker against all .py source files.

Execute the following bash command and report the results:

```bash
cd /Users/mikeodonnell/git/ODLabTracker && python dev/check_deps.py
```

After running, read and display the contents of `dependency_report.md`. Summarize:
1. Overall status (PASS or BLOCKED)
2. Any platform-specific imports that would block a commit
3. Any undeclared imports (warnings)
4. Any unused declared dependencies (warnings)

If there are issues, suggest specific fixes (e.g., remove the import, add the package to pyproject.toml, or refactor to avoid a platform-specific call).
