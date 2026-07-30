#!/usr/bin/env python3

import argparse
import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox


GRID = 28
CELL = 10
CANVAS = GRID * CELL


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def write_safetensors_x(path: pathlib.Path, values: list[float]) -> None:
    data = struct.pack("<" + "f" * len(values), *values)
    header = {
        "x": {
            "dtype": "F32",
            "shape": [1, len(values)],
            "data_offsets": [0, len(data)],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


class MnistDrawApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.grid = [0.0] * (GRID * GRID)
        self.last_point: tuple[float, float] | None = None

        self.root = tk.Tk()
        self.root.title("Sandy MNIST CPU Runner")
        self.canvas = tk.Canvas(self.root, width=CANVAS, height=CANVAS, bg="black")
        self.canvas.pack()

        buttons = tk.Frame(self.root)
        buttons.pack(fill=tk.X)
        tk.Button(buttons, text="Run", command=self.run).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(buttons, text="Clear", command=self.clear).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.canvas.bind("<Button-1>", self.paint_start)
        self.canvas.bind("<B1-Motion>", self.paint_drag)
        self.canvas.bind("<ButtonRelease-1>", self.paint_end)
        self.cells = []
        for y in range(GRID):
            row = []
            for x in range(GRID):
                row.append(self.canvas.create_rectangle(
                    x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL,
                    outline="", fill="#000000"))
            self.cells.append(row)

    def clear(self) -> None:
        self.grid = [0.0] * (GRID * GRID)
        self.last_point = None
        self.canvas.delete("stroke")
        self.redraw()

    def paint_start(self, event) -> None:
        self.last_point = (float(event.x), float(event.y))
        self.paint_at(float(event.x), float(event.y))

    def paint_drag(self, event) -> None:
        x = float(event.x)
        y = float(event.y)
        if self.last_point is not None:
            last_x, last_y = self.last_point
            self.canvas.create_line(
                last_x,
                last_y,
                x,
                y,
                fill="white",
                width=CELL * 3,
                capstyle=tk.ROUND,
                smooth=True,
                tags="stroke",
            )
        self.last_point = (x, y)
        self.paint_at(x, y)

    def paint_end(self, _event) -> None:
        self.last_point = None

    def paint_at(self, px: float, py: float) -> None:
        gx = px / CELL
        gy = py / CELL
        radius = 2.0
        brush = radius * CELL
        self.canvas.create_oval(
            px - brush,
            py - brush,
            px + brush,
            py + brush,
            outline="",
            fill="white",
            tags="stroke",
        )
        for y in range(GRID):
            for x in range(GRID):
                dist = math.hypot(x + 0.5 - gx, y + 0.5 - gy)
                if dist <= radius:
                    value = max(0.0, 1.0 - dist / radius) ** 0.65
                    idx = y * GRID + x
                    self.grid[idx] = min(1.0, max(self.grid[idx], value))
        self.redraw()

    def redraw(self) -> None:
        for y in range(GRID):
            for x in range(GRID):
                v = int(max(0.0, min(1.0, self.grid[y * GRID + x])) * 255)
                color = f"#{v:02x}{v:02x}{v:02x}"
                self.canvas.itemconfig(self.cells[y][x], fill=color)
        self.canvas.tag_raise("stroke")
        self.root.update_idletasks()

    def run(self) -> None:
        runner = pathlib.Path(self.args.runner)
        if not runner.exists():
            messagebox.showerror(
                "Missing cpu_runner",
                f"{runner} does not exist.\nBuild it with: cmake --build build --target cpu_runner")
            return

        with tempfile.NamedTemporaryFile(prefix="sandy_mnist_input_", suffix=".safetensors", delete=False) as f:
            input_path = pathlib.Path(f.name)
        write_safetensors_x(input_path, self.grid)

        cmd = [
            str(runner),
            str(self.args.model),
            str(self.args.weights),
            str(input_path),
        ]
        print("running:", " ".join(cmd))
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            messagebox.showerror("Runner failed", f"cpu_runner exited with {result.returncode}")
            return

        self.root.destroy()

    def mainloop(self) -> None:
        self.root.mainloop()


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default=root / "build/test/cpu_runner")
    parser.add_argument("--model", default=root / "src/models/mnist.sandy.go")
    parser.add_argument("--weights", default=root / "experiments/mnist/mnist.safetensors")
    args = parser.parse_args()

    app = MnistDrawApp(args)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
