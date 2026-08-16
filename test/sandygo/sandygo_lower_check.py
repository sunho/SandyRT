#!/usr/bin/env python3

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


CHECK_RE = re.compile(r"^\s*//\s*(CHECK(?:(?:-NEXT)|(?:-NOT))?):\s*(.*)$")
RUN_RE = re.compile(r"^\s*//\s*RUN:\s*(.*)$")


def filecheck_regex(pattern):
    pieces = []
    pos = 0
    while pos < len(pattern):
        start = pattern.find("{{", pos)
        if start < 0:
            pieces.append(re.escape(pattern[pos:]))
            break
        pieces.append(re.escape(pattern[pos:start]))
        end = pattern.find("}}", start + 2)
        if end < 0:
            pieces.append(re.escape(pattern[start:]))
            break
        pieces.append(pattern[start + 2:end])
        pos = end + 2
    return re.compile("".join(pieces))


def parse_test(path):
    runs = []
    checks = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        run = RUN_RE.match(line)
        if run:
            runs.append((line_number, run.group(1)))
            continue
        check = CHECK_RE.match(line)
        if check:
            checks.append((line_number, check.group(1), check.group(2)))
    if not runs:
        runs.append((0, "%sandygo_lower %s"))
    return runs, checks


def substitute(command, path, dump_mid_ir):
    substitutions = {
        "%sandygo_lower": dump_mid_ir,
        "%dump_mid_ir": dump_mid_ir,
        "%s": str(path),
    }
    for key, value in substitutions.items():
        command = command.replace(key, shlex.quote(value))
    return re.sub(r"\|\s*FileCheck\b.*$", "", command).strip()


def run_command(command, cwd):
    proc = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = proc.stdout
    if proc.stderr:
        output += proc.stderr
    return proc.returncode, output


def check_output(path, checks, output):
    lines = output.splitlines()
    cursor = 0
    previous_match = None

    for check_line, kind, pattern in checks:
        regex = filecheck_regex(pattern)
        if kind == "CHECK-NEXT":
            line_index = 0 if previous_match is None else previous_match + 1
            if line_index >= len(lines) or not regex.search(lines[line_index]):
                actual = "<eof>" if line_index >= len(lines) else lines[line_index]
                return (
                    False,
                    f"{path}:{check_line}: CHECK-NEXT failed\n"
                    f"  expected: {pattern}\n"
                    f"  actual:   {actual}",
                )
            previous_match = line_index
            cursor = line_index + 1
            continue

        if kind == "CHECK-NOT":
            for line_index in range(cursor, len(lines)):
                if regex.search(lines[line_index]):
                    return (
                        False,
                        f"{path}:{check_line}: CHECK-NOT failed\n"
                        f"  pattern: {pattern}\n"
                        f"  line:    {lines[line_index]}",
                    )
            continue

        for line_index in range(cursor, len(lines)):
            if regex.search(lines[line_index]):
                previous_match = line_index
                cursor = line_index + 1
                break
        else:
            tail = "\n".join(lines[max(0, cursor - 2):cursor + 8])
            return (
                False,
                f"{path}:{check_line}: CHECK failed\n"
                f"  expected: {pattern}\n"
                f"  near:\n{tail}",
            )

    return True, ""


def run_test(path, args):
    runs, checks = parse_test(path)
    if not checks:
        return False, f"{path}: no CHECK directives"

    for run_line, raw_command in runs:
        command = substitute(raw_command, path, args.dump_mid_ir)
        code, output = run_command(command, args.cwd)
        if code != 0:
            return (
                False,
                f"{path}:{run_line}: RUN failed with exit code {code}\n"
                f"  command: {command}\n{output}",
            )
        ok, message = check_output(path, checks, output)
        if not ok:
            return False, message + f"\n\nFull output:\n{output}"
    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-mid-ir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--cwd", default=os.getcwd())
    args = parser.parse_args()

    tests = sorted(Path(args.test_dir).rglob("*.sandy.go"))
    if not tests:
        print(f"no sandygo lower tests found under {args.test_dir}", file=sys.stderr)
        return 1

    failed = []
    for test in tests:
        ok, message = run_test(test, args)
        if ok:
            print(f"PASS: {test}")
        else:
            print(f"FAIL: {test}")
            print(message)
            failed.append(test)

    print(f"{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
