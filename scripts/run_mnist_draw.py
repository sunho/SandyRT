#!/usr/bin/env python3

import argparse
import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile

try:
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError:
    QPointF = None
    Qt = None
    QColor = None
    QImage = None
    QPainter = None
    QPen = None
    QApplication = None
    QHBoxLayout = None
    QMessageBox = None
    QPushButton = None
    QVBoxLayout = None
    QWidget = object


GRID = 28
CELL = 20
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
        self.window = MnistWindow(self)

    def clear(self) -> None:
        self.grid = [0.0] * (GRID * GRID)
        self.window.canvas.clear()

    def run(self) -> None:
        runner = pathlib.Path(self.args.runner)
        if not runner.exists():
            QMessageBox.critical(
                self.window,
                "Missing cpu_runner",
                f"{runner} does not exist.\nBuild it with: cmake --build build --target cpu_runner",
            )
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
            QMessageBox.critical(self.window, "Runner failed", f"cpu_runner exited with {result.returncode}")
            return

        QApplication.instance().quit()


class DrawCanvas(QWidget):
    def __init__(self, app: MnistDrawApp):
        super().__init__()
        self.app = app
        self.last_point: tuple[float, float] | None = None
        self.image = QImage(CANVAS, CANVAS, QImage.Format.Format_RGB32)
        self.image.fill(QColor("black"))
        self.setFixedSize(CANVAS, CANVAS)
        self.setMouseTracking(True)

    def clear(self) -> None:
        self.last_point = None
        self.image.fill(QColor("black"))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.clamp_point(event.position())
        self.last_point = point
        self.paint_at(*point)

    def mouseMoveEvent(self, event) -> None:
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        point = self.clamp_point(event.position())
        if self.last_point is not None:
            self.paint_segment(*self.last_point, *point)
        else:
            self.paint_at(*point)
        self.last_point = point

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = None

    def clamp_point(self, point: QPointF) -> tuple[float, float]:
        return (
            max(0.0, min(float(CANVAS - 1), point.x())),
            max(0.0, min(float(CANVAS - 1), point.y())),
        )

    def paint_segment(self, x0: float, y0: float, x1: float, y1: float) -> None:
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("white"), CELL * 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(int(x0), int(y0), int(x1), int(y1))
        painter.end()

        distance = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(distance / (CELL / 2)))
        for i in range(steps + 1):
            t = i / steps
            self.mark_grid_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        self.update()

    def paint_at(self, px: float, py: float) -> None:
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor("white"))
        painter.setPen(Qt.PenStyle.NoPen)
        brush = CELL * 1.5
        painter.drawEllipse(int(px - brush), int(py - brush), int(brush * 2), int(brush * 2))
        painter.end()

        self.mark_grid_at(px, py)
        self.update()

    def mark_grid_at(self, px: float, py: float) -> None:
        gx = px / CELL
        gy = py / CELL
        radius = 1.7
        for y in range(GRID):
            for x in range(GRID):
                dist = math.hypot(x + 0.5 - gx, y + 0.5 - gy)
                if dist <= radius:
                    self.app.grid[y * GRID + x] = 1.0


class MnistWindow(QWidget):
    def __init__(self, app: MnistDrawApp):
        super().__init__()
        self.app = app
        self.setWindowTitle("Sandy MNIST CPU Runner")
        self.canvas = DrawCanvas(app)

        run_button = QPushButton("Run")
        run_button.clicked.connect(app.run)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(app.clear)

        buttons = QHBoxLayout()
        buttons.addWidget(run_button)
        buttons.addWidget(clear_button)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addLayout(buttons)
        self.setLayout(layout)


def main() -> int:
    if QApplication is None:
        print(
            "PySide6 is required for the Qt MNIST drawing app.\n"
            "Install it with: python3 -m pip install PySide6",
            file=sys.stderr,
        )
        return 1

    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default=root / "build/test/cpu_runner")
    parser.add_argument("--model", default=root / "src/models/mnist.sandy.go")
    parser.add_argument("--weights", default=root / "experiments/mnist/mnist.safetensors")
    args = parser.parse_args()

    qt_app = QApplication(sys.argv)
    app = MnistDrawApp(args)
    app.window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
