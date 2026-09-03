from __future__ import annotations

from matplotlib.pyplot import axis
from raman_paths import LIBRARY_DIR
from matplotlib.ticker import MultipleLocator

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QAbstractSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from raman_backend import (
    AnalysisResult,
    AnalysisSettings,
    calculate_ideal_h2_air_composition,
    detect_library_temperature_step,
    extract_phi_from_filename,
    find_phi_library,
    load_processed_file,
    normalize,
    read_library_metadata,
    run_complete_analysis,
)


APP_DIR = Path(__file__).resolve().parent


class AnalysisThread(QThread):
    progress = Signal(int, str)
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        experimental_file: Path,
        phi: float,
        settings: AnalysisSettings,
        parent=None,
    ) -> None:
        super().__init__(parent)

        self.experimental_file = experimental_file
        self.phi = phi
        self.settings = settings

    def run(self) -> None:
        try:
            result = run_complete_analysis(
                experimental_file=self.experimental_file,
                phi=self.phi,
                settings=self.settings,
                base_dir=APP_DIR,
                progress_callback=self._report_progress,
            )

            self.result_ready.emit(result)

        except Exception as error:
            self.failed.emit(str(error))

    def _report_progress(
        self,
        fraction: float,
        message: str,
    ) -> None:
        percent = int(
            round(float(np.clip(fraction, 0.0, 1.0)) * 100)
        )
        self.progress.emit(percent, message)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6.5, 4.2), tight_layout=True)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 260)

    def show_message(self, title: str, message: str) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.axis("off")
        axis.set_title(title, fontsize=15, pad=18)
        axis.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=11,
        )
        self.draw_idle()


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "—") -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.value_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class VisibleCheckBox(QAbstractButton):
    """Checkbox-style button with separate fonts for the box and label text."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("visibleCheckBox")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(38)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 3, 10, 3)
        row.setSpacing(9)

        # Keep the checkbox symbol large and visible.
        self.symbol_label = QLabel("☐")
        self.symbol_label.setObjectName("checkboxSymbol")
        self.symbol_label.setFixedWidth(26)
        self.symbol_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.symbol_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        symbol_font = QFont("Segoe UI Symbol", 16)
        symbol_font.setBold(False)
        self.symbol_label.setFont(symbol_font)

        # Style only the words with the same font as the rest of the app.
        self.text_label = QLabel(text)
        self.text_label.setObjectName("checkboxText")
        self.text_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.text_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        text_font = QFont("Segoe UI", 10)
        text_font.setBold(False)
        self.text_label.setFont(text_font)

        row.addWidget(self.symbol_label)
        row.addWidget(self.text_label, 1)

        self.toggled.connect(self._refresh_symbol)
        self._refresh_symbol(False)

    @Slot(bool)
    def _refresh_symbol(self, checked: bool) -> None:
        self.symbol_label.setText("☑" if checked else "☐")

class NoWheelSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons
        )

    def wheelEvent(self, event):
        event.ignore()

class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons
        )

    def wheelEvent(self, event):
        event.ignore()

class RamanWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.selected_file: Path | None = None
        self.detected_phi: float | None = None
        self.current_result: AnalysisResult | None = None
        self.analysis_thread: AnalysisThread | None = None
        self.grid_resample_message: str | None = None

        self.setWindowTitle("H₂ Raman Temperature Fitter")
        self.resize(1200, 800)
        self.setMinimumSize(900, 650)

        self._build_ui()
        self._apply_styles()
        self._connect_signals()
        self._update_composition()
        self._sync_generation_wavelength_range()
        self._update_library_status()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 18, 20, 20)
        root_layout.setSpacing(14)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        title = QLabel("H₂ Raman Temperature Fitter")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Select a processed spectrum, supply Phi, and run the automatic temperature fit."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text)
        header.addStretch()
        root_layout.addLayout(header)

        # A movable divider lets the user make the menu wider or the graph smaller.
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(content_splitter, 1)

        # ---------- Left controls ----------
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_scroll.setMinimumWidth(430)
        controls_scroll.setMaximumWidth(560)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        controls_layout.setSpacing(12)
        controls_scroll.setWidget(controls_widget)
        content_splitter.addWidget(controls_scroll)

        file_group = QGroupBox("1. Experimental file")
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("mutedLabel")
        self.select_file_button = QPushButton("Select processed Raman file")
        self.select_file_button.setObjectName("secondaryButton")
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_file_button)

        column_form = QFormLayout()

        self.wavelength_column_input = QLineEdit()
        self.wavelength_column_input.setPlaceholderText("Enter column number")
        self.wavelength_column_input.setValidator(
            QIntValidator(1, 999, self)
        )
        self.wavelength_column_input.setToolTip(
            "Enter the column containing wavelength in nanometers."
        )

        self.intensity_column_input = QLineEdit()
        self.intensity_column_input.setPlaceholderText("Enter column number")
        self.intensity_column_input.setValidator(
            QIntValidator(1, 999, self)
        )
        self.intensity_column_input.setToolTip(
            "Enter the column containing experimental intensity."
        )

        column_form.addRow(
            "Wavelength column",
            self.wavelength_column_input,
        )

        column_form.addRow(
            "Intensity column",
            self.intensity_column_input,
        )

        file_layout.addLayout(column_form)

        column_note = QLabel(
            "Column numbering starts at 1, like Excel. "
        )   
        column_note.setWordWrap(True)
        column_note.setObjectName("mutedLabel")
        file_layout.addWidget(column_note)

        controls_layout.addWidget(file_group)

        phi_group = QGroupBox("2. Equivalence ratio")
        phi_layout = QVBoxLayout(phi_group)

        # Show the Phi found in the filename, then let the user edit the value
        # that will actually be used for composition, generation, and fitting.
        self.detected_phi_label = QLabel("Detected Phi: —")
        self.detected_phi_label.setObjectName("mutedLabel")

        phi_input_label = QLabel("Phi to use")
        self.phi_input = NoWheelDoubleSpinBox()
        self.phi_input.setRange(0.01, 20.0)
        self.phi_input.setDecimals(3)
        self.phi_input.setSingleStep(0.05)
        self.phi_input.setValue(1.6)
        self.phi_input.setToolTip(
            "The detected Phi is entered here automatically. You can type a different value."
        )

        phi_layout.addWidget(self.detected_phi_label)
        phi_layout.addWidget(phi_input_label)
        phi_layout.addWidget(self.phi_input)
        controls_layout.addWidget(phi_group)

        composition_group = QGroupBox("3. Calculated composition")
        composition_layout = QGridLayout(composition_group)
        self.composition_labels: dict[str, QLabel] = {}
        for index, species in enumerate(["H2", "N2", "H2O", "O2"]):
            name_label = QLabel(species.replace("2", "₂"))
            value_label = QLabel("—")
            value_label.setObjectName("compositionValue")
            self.composition_labels[species] = value_label
            row = index // 2
            column = (index % 2) * 2
            composition_layout.addWidget(name_label, row, column)
            composition_layout.addWidget(value_label, row, column + 1)
        controls_layout.addWidget(composition_group)

        # ---------- Fit settings ----------
        fit_group = QGroupBox("4. Fit settings")
        fit_layout = QFormLayout(fit_group)

        self.crop_start_input = NoWheelDoubleSpinBox()
        self.crop_start_input.setRange(1.0, 5000.0)
        self.crop_start_input.setDecimals(4)
        self.crop_start_input.setSingleStep(0.01)
        self.crop_start_input.setValue(297.0)

        self.crop_end_input = NoWheelDoubleSpinBox()
        self.crop_end_input.setRange(1.0, 5000.0)
        self.crop_end_input.setDecimals(4)
        self.crop_end_input.setSingleStep(0.01)
        self.crop_end_input.setValue(299.5)

        self.shift_checkbox = VisibleCheckBox("Allow small wavelength shift")
        self.shift_checkbox.setChecked(True)

        fit_layout.addRow("Fit wavelength start (nm)", self.crop_start_input)
        fit_layout.addRow("Fit wavelength end (nm)", self.crop_end_input)
        fit_layout.addRow(self.shift_checkbox)
        controls_layout.addWidget(fit_group)

        # ---------- Automatic library generation ----------
        generation_group = QGroupBox("5. Automatic library generation")
        generation_layout = QFormLayout(generation_group)

        generation_note = QLabel(
            "The program reuses an exact matching library. If none exists, "
            "it automatically generates one using these settings."
        )
        generation_note.setWordWrap(True)
        generation_note.setObjectName("mutedLabel")
        generation_layout.addRow(generation_note)

        self.excitation_input = NoWheelDoubleSpinBox()
        self.excitation_input.setRange(1.0, 5000.0)
        self.excitation_input.setDecimals(4)
        self.excitation_input.setSingleStep(0.1)
        self.excitation_input.setValue(266.0)

        self.gen_min_input = NoWheelSpinBox()
        self.gen_min_input.setRange(15, 10000)
        self.gen_min_input.setSingleStep(50)
        self.gen_min_input.setValue(1600)

        self.gen_max_input = NoWheelSpinBox()
        self.gen_max_input.setRange(15, 10000)
        self.gen_max_input.setSingleStep(50)
        self.gen_max_input.setValue(3000)

        self.gen_step_input = NoWheelSpinBox()
        self.gen_step_input.setRange(1, 100)
        self.gen_step_input.setValue(5)

        self.gen_wave_start_input = NoWheelDoubleSpinBox()
        self.gen_wave_start_input.setRange(1.0, 5000.0)
        self.gen_wave_start_input.setDecimals(4)
        self.gen_wave_start_input.setSingleStep(0.01)
        self.gen_wave_start_input.setValue(296.0)

        self.gen_wave_end_input = NoWheelDoubleSpinBox()
        self.gen_wave_end_input.setRange(1.0, 5000.0)
        self.gen_wave_end_input.setDecimals(4)
        self.gen_wave_end_input.setSingleStep(0.01)
        self.gen_wave_end_input.setValue(300.2)

        self.grid_increment_combo = NoWheelComboBox()
        self.grid_increment_combo.addItem("0.01 nm", 0.01)
        self.grid_increment_combo.addItem("0.001 nm", 0.001)
        self.grid_increment_combo.addItem("0.0001 nm", 0.0001)
        self.grid_increment_combo.setCurrentIndex(0)

        generation_layout.addRow("Excitation wavelength (nm)", self.excitation_input)
        generation_layout.addRow("Minimum temperature (K)", self.gen_min_input)
        generation_layout.addRow("Maximum temperature (K)", self.gen_max_input)
        generation_layout.addRow("Temperature step (K)", self.gen_step_input)
        generation_layout.addRow(
            "Generation wavelength start (nm)",
            self.gen_wave_start_input,
        )
        generation_layout.addRow(
            "Generation wavelength end (nm)",
            self.gen_wave_end_input,
        )
        generation_layout.addRow("Wavelength grid increment", self.grid_increment_combo)
        controls_layout.addWidget(generation_group)

        library_group = QGroupBox("6. Library status")
        library_layout = QVBoxLayout(library_group)
        self.library_status_label = QLabel("No Phi selected")
        self.library_status_label.setWordWrap(True)
        library_layout.addWidget(self.library_status_label)
        controls_layout.addWidget(library_group)

        self.run_button = QPushButton("Run temperature fit")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumHeight(48)
        self.run_button.setEnabled(False)
        controls_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        controls_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("statusLabel")
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

        # ---------- Right results ----------
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(12)
        results_widget.setMinimumWidth(420)
        content_splitter.addWidget(results_widget)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([460, 720])

        metric_layout = QHBoxLayout()
        metric_layout.setSpacing(10)
        self.temperature_card = MetricCard("Temperature")
        self.shift_card = MetricCard("Wavelength shift")
        self.fwhm_card = MetricCard("Gaussian FWHM")
        self.error_card = MetricCard("Fit error")
        for card in [
            self.temperature_card,
            self.shift_card,
            self.fwhm_card,
            self.error_card,
        ]:
            metric_layout.addWidget(card)
        results_layout.addLayout(metric_layout)

        self.tabs = QTabWidget()
        self.fit_canvas = PlotCanvas()
        self.error_canvas = PlotCanvas()
        self.residual_canvas = PlotCanvas()
        self.tabs.addTab(self.fit_canvas, "Spectrum")
        self.tabs.addTab(self.error_canvas, "Temperature error")
        self.tabs.addTab(self.residual_canvas, "Residual Plot")
        results_layout.addWidget(self.tabs, 1)

        button_row = QHBoxLayout()
        self.save_results_button = QPushButton("Save fit results CSV")
        self.save_results_button.setObjectName("secondaryButton")
        self.save_results_button.setEnabled(False)
        self.library_path_label = QLabel("Library: —")
        self.library_path_label.setObjectName("mutedLabel")
        self.library_path_label.setWordWrap(True)
        button_row.addWidget(self.library_path_label, 1)
        button_row.addWidget(self.save_results_button)
        results_layout.addLayout(button_row)

        self.fit_canvas.show_message(
            "Spectrum preview",
            "Select a processed Raman file to preview the experimental spectrum.",
        )
        self.error_canvas.show_message(
            "Temperature error",
            "The error curve will appear after the fit finishes.",
        )
        self.residual_canvas.show_message(
            "Residual plot",
            "The residual plot will appear after the fit finishes.",
        )

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f6f8;
                color: #17202a;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 13px;
            }
            QLabel#pageTitle {
                font-size: 28px;
                font-weight: 700;
                color: #102a43;
            }
            QLabel#pageSubtitle, QLabel#mutedLabel {
                color: #627d98;
            }
            QLabel#statusLabel {
                color: #486581;
                padding: 4px;
            }
            QLabel#compositionValue {
                font-weight: 700;
                color: #0b7285;
                min-width: 76px;
            }
            QGroupBox {
                background: white;
                border: 1px solid #d9e2ec;
                border-radius: 10px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            QFrame#metricCard {
                background: white;
                border: 1px solid #d9e2ec;
                border-radius: 10px;
            }
            QLabel#metricTitle {
                color: #627d98;
                font-size: 12px;
            }
            QLabel#metricValue {
                color: #102a43;
                font-size: 22px;
                font-weight: 700;
            }
            QPushButton {
                border-radius: 8px;
                padding: 9px 14px;
                font-weight: 600;
            }
            QPushButton#primaryButton {
                background: #0b7285;
                color: white;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background: #08626f;
            }
            QPushButton#primaryButton:disabled {
                background: #bcccdc;
                color: #f0f4f8;
            }
            QPushButton#secondaryButton {
                background: white;
                color: #0b7285;
                border: 1px solid #9fb3c8;
            }
            QPushButton#secondaryButton:hover {
                background: #e6fffa;
            }
            QAbstractButton#visibleCheckBox {
                background: white;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
            }
            QAbstractButton#visibleCheckBox:hover {
                background: #f0fffc;
                border: 1px solid #829ab1;
            }
            QAbstractButton#visibleCheckBox:checked {
                background: #e6fffa;
                border: 1px solid #0b7285;
            }
            QAbstractButton#visibleCheckBox:focus {
                border: 2px solid #0b7285;
            }
            QLabel#checkboxSymbol {
                background: transparent;
                color: #000000;
                border: none;
                padding: 0px;
            }
            QLabel#checkboxText {
                background: transparent;
                color: #17202a;
                border: none;
                padding: 0px;
            }
            QGroupBox#advancedPanel {
                margin-top: 0px;
            }
            QDoubleSpinBox, QSpinBox, QComboBox {
                background: white;
                border: 1px solid #bcccdc;
                border-radius: 6px;
                padding: 6px;
            }
            QProgressBar {
                background: white;
                border: 1px solid #bcccdc;
                border-radius: 7px;
                text-align: center;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background: #0b7285;
                border-radius: 6px;
            }
            QTabWidget::pane {
                background: white;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #e9eef3;
                padding: 9px 18px;
                margin-right: 2px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }
            QTabBar::tab:selected {
                background: white;
                color: #0b7285;
                font-weight: 700;
            }
            """
        )

    def _connect_signals(self) -> None:
        self.select_file_button.clicked.connect(self.select_file)
        self.wavelength_column_input.textChanged.connect(
            lambda _text: self.refresh_column_preview()
        )
        self.intensity_column_input.textChanged.connect(
            lambda _text: self.refresh_column_preview()
        )
        self.phi_input.valueChanged.connect(self._phi_changed)
        self.run_button.clicked.connect(self.start_analysis)
        self.save_results_button.clicked.connect(self.save_results)
        self.crop_start_input.valueChanged.connect(self._fit_wavelength_changed)
        self.crop_end_input.valueChanged.connect(self._fit_wavelength_changed)
        self.shift_checkbox.toggled.connect(
            lambda _checked: self._fit_wavelength_changed()
        )
        self.grid_increment_combo.currentIndexChanged.connect(
            lambda _index: self._generation_setting_changed()
        )
        self.excitation_input.valueChanged.connect(
            lambda _value: self._generation_setting_changed()
        )
        self.gen_min_input.valueChanged.connect(
            lambda _value: self._generation_setting_changed()
        )
        self.gen_max_input.valueChanged.connect(
            lambda _value: self._generation_setting_changed()
        )
        self.gen_step_input.valueChanged.connect(
            lambda _value: self._generation_setting_changed()
        )
        self.gen_wave_start_input.valueChanged.connect(
            lambda _value: self._generation_setting_changed()
        )
        self.gen_wave_end_input.valueChanged.connect(
            lambda _value: self._generation_setting_changed()
        )

    @Slot()
    def _generation_setting_changed(self) -> None:
        self._update_library_status()

    @Slot()
    def _fit_wavelength_changed(self) -> None:
        self._sync_generation_wavelength_range()
        self._update_library_status()

    def _sync_generation_wavelength_range(self, *, force: bool = False) -> None:
        """Ensure the generation grid covers the selected fit interval."""
        fit_start = float(self.crop_start_input.value())
        fit_end = float(self.crop_end_input.value())
        if fit_end <= fit_start:
            return

        current_start = float(self.gen_wave_start_input.value())
        current_end = float(self.gen_wave_end_input.value())
        if not force and current_start <= fit_start and current_end >= fit_end:
            return

        grid_step = float(self.grid_increment_combo.currentData() or 0.01)
        padding = 0.30 if self.shift_checkbox.isChecked() else 0.20
        generated_start = np.floor((fit_start - padding) / grid_step) * grid_step
        generated_end = np.ceil((fit_end + padding) / grid_step) * grid_step

        self.gen_wave_start_input.blockSignals(True)
        self.gen_wave_end_input.blockSignals(True)
        self.gen_wave_start_input.setValue(max(1.0, generated_start))
        self.gen_wave_end_input.setValue(generated_end)
        self.gen_wave_start_input.blockSignals(False)
        self.gen_wave_end_input.blockSignals(False)

    def _adapt_fit_range_to_experiment(self, exp_df: pd.DataFrame) -> None:
        """Use the experimental wavelength range when defaults do not overlap."""
        wavelengths = exp_df["exp_wavelength_nm"].to_numpy(dtype=float)
        wavelengths = wavelengths[np.isfinite(wavelengths)]
        if wavelengths.size < 10:
            return

        fit_start = float(self.crop_start_input.value())
        fit_end = float(self.crop_end_input.value())
        points_in_fit = np.count_nonzero(
            (wavelengths >= fit_start) & (wavelengths <= fit_end)
        )
        if points_in_fit >= 10:
            return

        exp_start = float(np.min(wavelengths))
        exp_end = float(np.max(wavelengths))
        self.crop_start_input.blockSignals(True)
        self.crop_end_input.blockSignals(True)
        self.crop_start_input.setValue(exp_start)
        self.crop_end_input.setValue(exp_end)
        self.crop_start_input.blockSignals(False)
        self.crop_end_input.blockSignals(False)
        self._sync_generation_wavelength_range(force=True)

    @Slot()
    def select_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select experimental Raman file",
            str(APP_DIR),
            (
                "Raman data (*.xls *.xlsx *.csv *.txt *.dat);;"
                "All files (*.*)"
            ),
        )

        if not filename:
            return

        self.selected_file = Path(filename)
        self.file_label.setText(str(self.selected_file))

        self.detected_phi = extract_phi_from_filename(
            self.selected_file.name
        )

        if self.detected_phi is not None:
            self.detected_phi_label.setText(
                f"Detected Phi: {self.detected_phi:g}"
            )

            self.phi_input.blockSignals(True)
            self.phi_input.setValue(self.detected_phi)
            self.phi_input.blockSignals(False)

        else:
            self.detected_phi_label.setText(
                "Detected Phi: not found"
            )

        # Clear the column boxes every time a new file is selected.
        self.wavelength_column_input.blockSignals(True)
        self.intensity_column_input.blockSignals(True)

        self.wavelength_column_input.clear()
        self.intensity_column_input.clear()

        self.wavelength_column_input.blockSignals(False)
        self.intensity_column_input.blockSignals(False)

        # Do not allow analysis until both valid columns are entered.
        self.run_button.setEnabled(False)

        self.status_label.setText(
            "File selected. Enter the wavelength and "
            "intensity column numbers."
        )

        self.fit_canvas.show_message(
            "Enter column numbers",
            "Enter the wavelength and intensity columns "
            "to load the experimental preview.",
        )

        self._update_composition()
        self._update_library_status()


    @Slot()
    def refresh_column_preview(self) -> None:
        if self.selected_file is None:
            return

        wavelength_text = (
            self.wavelength_column_input.text().strip()
        )
        intensity_text = (
            self.intensity_column_input.text().strip()
        )

        # Wait until both boxes contain a number.
        if not wavelength_text or not intensity_text:
            self.run_button.setEnabled(False)

            self.status_label.setText(
                "Enter both the wavelength and intensity "
                "column numbers."
            )

            self.fit_canvas.show_message(
                "Enter column numbers",
                "The preview will appear after both "
                "column numbers are entered.",
            )
            return

        wavelength_column = int(wavelength_text)
        intensity_column = int(intensity_text)

        if wavelength_column == intensity_column:
            self.run_button.setEnabled(False)

            self.status_label.setText(
                "Wavelength and intensity must use "
                "different columns."
            )

            self.fit_canvas.show_message(
                "Invalid column selection",
                "Choose different columns for wavelength "
                "and intensity.",
            )
            return

        try:
            theory_df, exp_df = load_processed_file(
                self.selected_file,
                wavelength_column=wavelength_column,
                intensity_column=intensity_column,
            )

            self._adapt_fit_range_to_experiment(exp_df)
            self._plot_preview(theory_df, exp_df)

            self.status_label.setText(
                f"Preview loaded using wavelength column "
                f"{wavelength_column} and intensity column "
                f"{intensity_column}."
            )

            self.run_button.setEnabled(True)

        except Exception as error:
            self.run_button.setEnabled(False)

            self.status_label.setText(str(error))

            self.fit_canvas.show_message(
                "Could not load selected columns",
                str(error),
            )

    def _plot_preview(self, theory_df: pd.DataFrame, exp_df: pd.DataFrame) -> None:
        self.fit_canvas.figure.clear()
        axis = self.fit_canvas.figure.add_subplot(111)
        axis.plot(
            exp_df["exp_wavelength_nm"],
            exp_df["exp_intensity"],
            label="Experimental",
        )
        if not theory_df.empty:
            axis.plot(
                theory_df["theory_wavelength_nm"],
                theory_df["theory_intensity"],
                label="Provided theory curve",
                alpha=0.75,
            )
        axis.axvspan(
            self.crop_start_input.value(),
            self.crop_end_input.value(),
            alpha=0.15,
            label="Fit region",
        )
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Intensity")
        axis.set_title("Uploaded spectrum preview")
        axis.grid(True, alpha=0.25)
        axis.legend()
        self.fit_canvas.draw_idle()

    @Slot(float)
    def _phi_changed(self, _value: float) -> None:
        self._update_composition()
        self._update_library_status()

    def _current_phi(self) -> float:
        # The editable box is always the value used by the program.
        # A detected filename value is copied into this box automatically.
        return float(self.phi_input.value())

    def _update_composition(self) -> None:
        try:
            _, _, composition = calculate_ideal_h2_air_composition(self._current_phi())
            for species in ["H2", "N2", "H2O", "O2"]:
                self.composition_labels[species].setText(
                    f"{composition[species]:.6f}"
                )
        except Exception:
            for label in self.composition_labels.values():
                label.setText("—")

    def _update_library_status(self) -> None:
        try:
            phi = self._current_phi()
            requested_temp_min = int(self.gen_min_input.value())
            requested_temp_max = int(self.gen_max_input.value())
            requested_temp_step = int(self.gen_step_input.value())
            requested_grid_step = float(self.grid_increment_combo.currentData())
            requested_excitation = float(self.excitation_input.value())
            requested_wave_start = float(self.gen_wave_start_input.value())
            requested_wave_end = float(self.gen_wave_end_input.value())
            _, _, composition = calculate_ideal_h2_air_composition(phi)

            requested_summary = (
                f"Excitation: {requested_excitation:g} nm\n"
                f"Generation wavelength range: "
                f"{requested_wave_start:g}–{requested_wave_end:g} nm\n"
                f"Wavelength grid step: {requested_grid_step:g} nm\n"
                f"Generation temperature range: "
                f"{requested_temp_min}–{requested_temp_max} K\n"
                f"Temperature step: {requested_temp_step} K"
            )

            library = find_phi_library(
                phi,
                APP_DIR,
                excitation_nm=requested_excitation,
                grid_start_nm=requested_wave_start,
                grid_end_nm=requested_wave_end,
                grid_step_nm=requested_grid_step,
                requested_temp_min=requested_temp_min,
                requested_temp_max=requested_temp_max,
                requested_temp_step=requested_temp_step,
                composition=composition,
            )

            if library is None:
                self.library_status_label.setText(
                    f"No matching Phi {phi:g} library was found.\n\n"
                    f"The next run will automatically generate one using:\n"
                    f"{requested_summary}"
                )
                return

            metadata = read_library_metadata(library)
            self.library_status_label.setText(
                f"Matching library: {library.name}\n\n"
                f"Phi: {float(metadata['phi']):g}\n"
                f"Excitation: {float(metadata['excitation_nm']):g} nm\n"
                f"Wavelength range: "
                f"{float(metadata['grid_start_nm']):g}–"
                f"{float(metadata['grid_end_nm']):g} nm\n"
                f"Wavelength grid step: "
                f"{float(metadata['grid_step_nm']):g} nm\n"
                f"Wavelength points: {int(metadata['wavelength_count']):,}\n"
                f"Available temperature range: "
                f"{float(metadata['temperature_min_K']):g}–"
                f"{float(metadata['temperature_max_K']):g} K\n"
                f"Temperature step: "
                f"{float(metadata['temperature_step_K']):g} K\n"
                f"The fit will use {requested_temp_min}–"
                f"{requested_temp_max} K."
            )

        except Exception as error:
            self.library_status_label.setText(str(error))

    def _collect_settings(self) -> AnalysisSettings:
        wavelength_text = (
            self.wavelength_column_input.text().strip()
        )

        intensity_text = (
            self.intensity_column_input.text().strip()
        )

        if not wavelength_text or not intensity_text:
            raise ValueError(
                "Enter the wavelength and intensity "
                "column numbers."
            )

        wavelength_column = int(wavelength_text)
        intensity_column = int(intensity_text)

        if wavelength_column == intensity_column:
            raise ValueError(
                "Wavelength and intensity must use "
                "different columns."
            )
        crop_start = float(self.crop_start_input.value())
        crop_end = float(self.crop_end_input.value())
        generation_wave_start = float(self.gen_wave_start_input.value())
        generation_wave_end = float(self.gen_wave_end_input.value())

        if crop_end <= crop_start:
            raise ValueError("Fit end must be greater than fit start.")
        if self.gen_max_input.value() < self.gen_min_input.value():
            raise ValueError(
                "Generation maximum temperature must be at least the minimum."
            )
        if generation_wave_end <= generation_wave_start:
            raise ValueError(
                "Generation wavelength end must be greater than its start."
            )
        if crop_start < generation_wave_start or crop_end > generation_wave_end:
            raise ValueError(
                "The generation wavelength range must fully cover the fit "
                "wavelength range. Expand the generation start/end values."
            )

        return AnalysisSettings(
            wavelength_column=wavelength_column,
            intensity_column=intensity_column,
            crop_start=crop_start,
            crop_end=crop_end,
            fit_temp_step=int(self.gen_step_input.value()),
            allow_shift=self.shift_checkbox.isChecked(),
            auto_generate_missing=True,
            regenerate_library=False,
            generation_temp_min=int(self.gen_min_input.value()),
            generation_temp_max=int(self.gen_max_input.value()),
            generation_temp_step=int(self.gen_step_input.value()),
            excitation_nm=float(self.excitation_input.value()),
            generation_grid_start_nm=generation_wave_start,
            generation_grid_end_nm=generation_wave_end,
            grid_step_nm=float(self.grid_increment_combo.currentData()),
        )

    @Slot()
    def start_analysis(self) -> None:
        if self.selected_file is None:
            QMessageBox.warning(
                self,
                "No file",
                "Select a processed Raman file first.",
            )
            return

        # Do not start another analysis while one is currently running.
        if (
            self.analysis_thread is not None
            and self.analysis_thread.isRunning()
        ):
            return

        try:
            phi = self._current_phi()
            requested_temp_min = int(self.gen_min_input.value())
            requested_temp_max = int(self.gen_max_input.value())
            requested_temp_step = int(self.gen_step_input.value())
            requested_grid_step = float(
                self.grid_increment_combo.currentData()
            )
            requested_excitation = float(self.excitation_input.value())
            requested_wave_start = float(self.gen_wave_start_input.value())
            requested_wave_end = float(self.gen_wave_end_input.value())

            _, _, composition = calculate_ideal_h2_air_composition(phi)

            existing_library = find_phi_library(
                phi,
                APP_DIR,
                excitation_nm=requested_excitation,
                grid_start_nm=requested_wave_start,
                grid_end_nm=requested_wave_end,
                grid_step_nm=requested_grid_step,
                requested_temp_min=requested_temp_min,
                requested_temp_max=requested_temp_max,
                requested_temp_step=requested_temp_step,
                composition=composition,
            )

            settings = self._collect_settings()

        except Exception as error:
            QMessageBox.warning(
                self,
                "Invalid settings",
                str(error),
            )
            return

        self.current_result = None
        self.grid_resample_message = None

        self.save_results_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")
        self.select_file_button.setEnabled(False)

        self.progress_bar.setValue(0)
        self.status_label.setText("Starting analysis…")

        self.temperature_card.set_value("—")
        self.shift_card.set_value("—")
        self.fwhm_card.set_value("—")
        self.error_card.set_value("—")

        # Create a completely new thread for every run.
        self.analysis_thread = AnalysisThread(
            experimental_file=self.selected_file,
            phi=phi,
            settings=settings,
            parent=self,
        )

        self.analysis_thread.progress.connect(
            self._analysis_progress
        )
        self.analysis_thread.result_ready.connect(
            self._analysis_finished
        )
        self.analysis_thread.failed.connect(
            self._analysis_failed
        )
        self.analysis_thread.finished.connect(
            self._analysis_thread_finished
        )

        self.analysis_thread.start()

    @Slot(int, str)
    def _analysis_progress(
        self,
        percent: int,
        message: str,
    ) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

        # Keep the grid-resampling message because later fitting
        # progress messages would otherwise immediately replace it.
        if "wavelength grid resampled" in message.lower():
            self.grid_resample_message = message
            self.library_path_label.setText(message)

    @Slot(object)
    def _analysis_finished(self, result: AnalysisResult) -> None:
        self.current_result = result
        self.progress_bar.setValue(100)
        self.status_label.setText("Analysis complete")
        self.temperature_card.set_value(f"{result.best_temperature} K")
        self.shift_card.set_value(f"{result.best_shift:.4f} nm")
        self.fwhm_card.set_value(f"{result.best_broadening:.4f} nm")
        self.error_card.set_value(f"{result.best_error:.6f}")
        library_text = f"Library: {result.library_path}"
        
        if self.grid_resample_message is not None:
            library_text += (
                "\n\nWorking wavelength grid:\n"
                + self.grid_resample_message
            )

        self.library_path_label.setText(library_text)
        self.save_results_button.setEnabled(True)
        self._plot_result(result)
        self._update_library_status()

        if np.isclose(result.best_temperature, result.temps.min()) or np.isclose(
            result.best_temperature, result.temps.max()
        ):
            QMessageBox.warning(
                self,
                "Temperature range warning",
                "The best temperature is at the edge of the library range. "
                "Expand the manual temperature range and run the fit again.",
            )

    @Slot(str)
    def _analysis_failed(self, message: str) -> None:
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis failed")
        QMessageBox.critical(
        self,
        "Raman analysis error",
        message,
    )

    @Slot()
    def _analysis_thread_finished(self) -> None:
        finished_thread = self.sender()

        # Clear the current thread only when this is the active one.
        if self.analysis_thread is finished_thread:
            self.analysis_thread = None

        self.run_button.setEnabled(True)
        self.run_button.setText("Run temperature fit")
        self.select_file_button.setEnabled(True)

        if self.current_result is not None:
            self.status_label.setText(
                "Analysis complete — ready to run again"
            )

        if finished_thread is not None:
            finished_thread.deleteLater()

    def _plot_result(self, result: AnalysisResult) -> None:
        self.fit_canvas.figure.clear()
        axis = self.fit_canvas.figure.add_subplot(111)
        axis.plot(
            result.experimental_wavelengths_crop,
            normalize(result.experimental_intensity_crop),
            label="Experimental",
            linewidth=2,
        )
        axis.plot(
            result.experimental_wavelengths_crop,
            normalize(result.best_simulation),
            label=f"Best simulation: {result.best_temperature} K",
            linewidth=2,
        )
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Normalized intensity")
        axis.set_title(f"Best fit for Phi {result.phi:g}")
        axis.grid(True, alpha=0.25)
        axis.legend()
        self.fit_canvas.draw_idle()

        best_by_temp = (
            result.fit_results.sort_values("Error")
            .groupby("Temperature (K)", as_index=False)
            .first()
            .sort_values("Temperature (K)")
        )

        self.error_canvas.figure.clear()
        error_axis = self.error_canvas.figure.add_subplot(111)
        error_axis.plot(
            best_by_temp["Temperature (K)"],
            best_by_temp["Error"],
        )
        error_axis.axvline(
            result.best_temperature,
            linestyle="--",
            label=f"Best: {result.best_temperature} K",
        )
        error_axis.set_xlabel("Temperature (K)")
        error_axis.set_ylabel("Best mean squared error")
        error_axis.set_title("Temperature error curve")
        error_axis.grid(True, alpha=0.25)
        error_axis.legend()
        self.error_canvas.draw_idle()
        self.tabs.setCurrentWidget(self.fit_canvas)

        # Convert the result arrays to NumPy arrays.
        wavelengths = np.asarray(
            result.experimental_wavelengths_crop,
            dtype=float,
        )
        experimental = np.asarray(
            result.experimental_intensity_crop,
            dtype=float,
        )
        simulation = np.asarray(
            result.best_simulation,
            dtype=float,
        )

        valid = (
            np.isfinite(wavelengths)
            & np.isfinite(experimental)
            & np.isfinite(simulation)
        )

        wavelengths = wavelengths[valid]
        experimental = experimental[valid]
        simulation = simulation[valid]

        # Normalize both spectra before calculating the residual.
        experimental_normalized = normalize(experimental)
        simulation_normalized = normalize(simulation)

        # Experimental minus simulated intensity.
        residuals = experimental_normalized - simulation_normalized

        # Magnify and vertically offset the residual for display only.
        residual_scale = 1.0
        residual_offset = -0.08
        display_residuals = residual_offset + residual_scale * residuals

        self.residual_canvas.figure.clear()
        axis = self.residual_canvas.figure.add_subplot(111)

        # Experimental spectrum.
        axis.plot(
            wavelengths,
            experimental_normalized,
            linewidth=1.3,
            label="Experimental",
        )
        
        # Best-fit simulated spectrum.
        axis.plot(
            wavelengths,
            simulation_normalized,
            linewidth=1.5,
            label="Best-fit simulation",
        )

        # Residual trace below the spectra.
        axis.plot(
            wavelengths,
            display_residuals,
            linewidth=1.0,
            label="Residual",
        )

        # Dashed reference line corresponding to zero residual.
        axis.axhline(
            residual_offset,
            linestyle="--",
            linewidth=1.0,
        )

        axis.text(
            wavelengths.min(),
            residual_offset - 0.025,
            "Residual",
            ha="left",
            va="top",
        )

        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Normalized intensity")
        axis.set_title(
            f"Best Fit and Residuals at {result.best_temperature} K"
        )

        axis.set_xlim(wavelengths.min(), wavelengths.max())
        axis.set_ylim(
            min(-0.15, float(np.min(display_residuals)) - 0.02),
            1.05,
        )
        axis.yaxis.set_major_locator(MultipleLocator(0.10))
        axis.yaxis.set_minor_locator(MultipleLocator(0.05))
        axis.grid(True, which="major", alpha=0.25)
        axis.grid(True, which="minor", alpha=0.10)
        axis.tick_params(axis="y", which="major", length=6)
        axis.tick_params(axis="y", which="minor", length=3)

        axis.grid(True, alpha=0.20)
        axis.legend()
        self.residual_canvas.figure.tight_layout()
        self.residual_canvas.draw_idle()



    @Slot()
    def save_results(self) -> None:
        if self.current_result is None:
            return

        default_name = (
            f"phi_{self.current_result.phi:g}_temperature_fit_results.csv"
        ).replace(".", "p", 1)
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save fit results",
            str(APP_DIR / default_name),
            "CSV files (*.csv)",
        )
        if not filename:
            return

        output_path = Path(filename)
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")

        result = self.current_result
        detailed = result.fit_results.sort_values("Error").copy()
        metadata = read_library_metadata(result.library_path)
        detailed.insert(0, "Phi", result.phi)
        detailed.insert(1, "Best overall temperature (K)", result.best_temperature)
        detailed.insert(2, "Best overall shift (nm)", result.best_shift)
        detailed.insert(3, "Best overall FWHM (nm)", result.best_broadening)
        detailed.insert(4, "Best overall error", result.best_error)
        detailed.insert(5, "Library filename", result.library_path.name)
        detailed.insert(6, "Excitation wavelength (nm)", metadata["excitation_nm"])
        detailed.insert(7, "Library wavelength start (nm)", metadata["grid_start_nm"])
        detailed.insert(8, "Library wavelength end (nm)", metadata["grid_end_nm"])
        detailed.insert(9, "Library grid step (nm)", metadata["grid_step_nm"])
        detailed.insert(10, "Library temperature minimum (K)", metadata["temperature_min_K"])
        detailed.insert(11, "Library temperature maximum (K)", metadata["temperature_max_K"])
        detailed.insert(12, "Library temperature step (K)", metadata["temperature_step_K"])
        detailed.to_csv(output_path, index=False)

        QMessageBox.information(
            self,
            "Results saved",
            f"Fit results were saved to:\n{output_path}",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name
        if (
            self.analysis_thread is not None
            and self.analysis_thread.isRunning()
        ):
            QMessageBox.warning(
                self,
                "Analysis running",
                "Wait for the current analysis to finish before closing the program.",
            )
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("H₂ Raman Temperature Fitter")
    app.setFont(QFont("Segoe UI", 10))

    window = RamanWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
