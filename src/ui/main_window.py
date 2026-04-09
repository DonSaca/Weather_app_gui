from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QDateEdit, QComboBox, QPushButton, QFrame, QMessageBox)
from PySide6.QtCore import QDate, Qt
from src.ui.workers import PredictionWorker
from src.ml.pipeline import WeatherModelWrapper

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PAP: Previsão Meteorológica - Lisboa")
        self.setStyleSheet("QMainWindow { background-color: #f3f4f6; font-family: 'Segoe UI', Arial; }")

        try:
            self.model_wrapper = WeatherModelWrapper()
        except Exception as e:
            QMessageBox.critical(self, "Erro no Sistema", f"Falha ao carregar modelos/scalers:\n{str(e)}")
            import sys; sys.exit(1)

        self.setup_ui()
        self.showMaximized() # Launch full-screen

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Left Column: Controls
        control_frame = QFrame()
        control_frame.setMaximumWidth(400)
        control_frame.setStyleSheet("background-color: white; border-radius: 12px; border: 1px solid #e5e7eb;")
        control_layout = QVBoxLayout(control_frame)
        
        lbl_title = QLabel("Menu")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; border: none; color: #374151;")
        
        self.date_edit = QDateEdit()
        # Cap selection to yesterday to prevent API gaps
        self.date_edit.setMaximumDate(QDate.currentDate().addDays(-1)) 
        self.date_edit.setDate(QDate.currentDate().addDays(-1))
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setStyleSheet("padding: 10px; border: 1px solid #d1d5db; border-radius: 6px; color: black; font-size: 14px;")
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "Transformer"])
        self.model_combo.setStyleSheet("padding: 10px; border: 1px solid #d1d5db; border-radius: 6px; color: black; font-size: 14px;")
        
        self.btn_predict = QPushButton("Executar Previsão")
        self.btn_predict.setStyleSheet("""
            QPushButton { background-color: #2563eb; color: white; border-radius: 6px; padding: 12px; font-weight: bold; font-size: 14px; }
            QPushButton:disabled { background-color: #93c5fd; }
            QPushButton:hover { background-color: #1d4ed8; }
        """)
        self.btn_predict.clicked.connect(self.run_prediction)

        control_layout.addWidget(lbl_title)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Data de Validação:"))
        control_layout.addWidget(self.date_edit)
        control_layout.addSpacing(10)
        control_layout.addWidget(QLabel("Selecionar Modelo:"))
        control_layout.addWidget(self.model_combo)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_predict)
        
        # Right Column: Hero Metrics
        hero_frame = QFrame()
        hero_frame.setStyleSheet("background-color: white; border-radius: 12px; border: 1px solid #e5e7eb;")
        hero_layout = QVBoxLayout(hero_frame)
        hero_layout.setAlignment(Qt.AlignCenter)
        
        self.lbl_hero_pred = QLabel("-- °C")
        self.lbl_hero_pred.setStyleSheet("font-size: 64px; font-weight: bold; color: #111827; border: none;")
        self.lbl_hero_pred.setAlignment(Qt.AlignCenter)

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)

        self.lbl_hero_actual = QLabel("Real: -- °C")
        self.lbl_hero_actual.setStyleSheet("font-size: 18px; color: #6b7280; border: none;")
        self.lbl_hero_actual.setAlignment(Qt.AlignCenter)
        
        self.lbl_hero_delta = QLabel("Erro: --")
        self.lbl_hero_delta.setStyleSheet("font-size: 16px; font-weight: bold; color: white; background-color: #9ca3af; border-radius: 12px; padding: 6px 16px;")
        self.lbl_hero_delta.setAlignment(Qt.AlignCenter)

        lbl_pred_title = QLabel("Máxima Prevista", alignment=Qt.AlignCenter)
        lbl_pred_title.setStyleSheet("font-size: 18px; color: #4b5563; border: none;")

        hero_layout.addWidget(lbl_pred_title)
        hero_layout.addWidget(self.lbl_hero_pred)
        hero_layout.addWidget(self.lbl_status)
        hero_layout.addSpacing(30)
        hero_layout.addWidget(self.lbl_hero_actual)
        hero_layout.addSpacing(10)
        hero_layout.addWidget(self.lbl_hero_delta, alignment=Qt.AlignHCenter)

        main_layout.addWidget(control_frame, 1)
        main_layout.addWidget(hero_frame, 3)

    def run_prediction(self):
        self.btn_predict.setEnabled(False)
        self.btn_predict.setText("A calcular...")
        self.lbl_status.setText("")
        
        target_date = self.date_edit.date().toString("yyyy-MM-dd")
        model_name = self.model_combo.currentText()
        
        self.worker = PredictionWorker(target_date, model_name, self.model_wrapper)
        self.worker.finished.connect(self.on_prediction_success)
        self.worker.error.connect(self.on_prediction_error)
        self.worker.start()

    def on_prediction_success(self, results):
        pred = results['predicted']
        self.lbl_hero_pred.setText(f"{pred} °C")
        self.lbl_hero_actual.setText(f"Real: {results['actual']} °C")
        
        delta = results['delta']
        self.lbl_hero_delta.setText(f"Erro: ±{delta} °C")
        
        # Weather Status Styling
        if pred > 25.0:
            self.lbl_status.setText("Dia Quente 🔥")
            self.lbl_status.setStyleSheet("font-size: 20px; font-weight: bold; color: #ea580c; border: none;")
        elif pred >= 15.0:
            self.lbl_status.setText("Razoável 🌤️")
            self.lbl_status.setStyleSheet("font-size: 20px; font-weight: bold; color: #ca8a04; border: none;")
        else:
            self.lbl_status.setText("Frio ❄️")
            self.lbl_status.setStyleSheet("font-size: 20px; font-weight: bold; color: #0284c7; border: none;")

        # Accuracy Badge Styling
        if delta <= 1.5:
            color = "#10b981"
        elif delta <= 3.0:
            color = "#f59e0b"
        else:
            color = "#ef4444"
            
        self.lbl_hero_delta.setStyleSheet(f"font-size: 16px; font-weight: bold; color: white; background-color: {color}; border-radius: 12px; padding: 6px 14px;")
        
        self.reset_button()

    def on_prediction_error(self, err_msg):
        QMessageBox.warning(self, "Aviso", err_msg)
        self.reset_button()

    def reset_button(self):
        self.btn_predict.setEnabled(True)
        self.btn_predict.setText("Executar Previsão")