import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from teamwork_s2s.src.core.logger import Logger

@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "metrics_test.db")

def test_logger_db_creation(temp_db):
    logger = Logger(db_path=temp_db)
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fer_logs'")
    assert cursor.fetchone() is not None
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='kpi_benchmarks'")
    assert cursor.fetchone() is not None
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_metrics'")
    assert cursor.fetchone() is not None
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_reports'")
    assert cursor.fetchone() is not None
    conn.close()

def test_logger_kpi_monitoring(temp_db):
    logger = Logger(db_path=temp_db)
    
    mock_vm = MagicMock()
    mock_vm.percent = 40.0
    mock_du = MagicMock()
    mock_du.percent = 20.0
    
    with patch("psutil.cpu_percent", return_value=12.5), \
         patch("psutil.virtual_memory", return_value=mock_vm), \
         patch("psutil.disk_usage", return_value=mock_du):
         
        kpis = logger.measure_hardware_kpis()
        assert kpis["cpu_percent"] == 12.5
        assert kpis["ram_percent"] == 40.0
        assert kpis["disk_percent"] == 20.0
        
    # Verify insertion
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT cpu_percent, ram_percent, disk_percent FROM kpi_benchmarks")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == 12.5
    assert row[1] == 40.0
    assert row[2] == 20.0
    conn.close()

def test_logger_report_generation(temp_db):
    logger = Logger(db_path=temp_db)
    logger.write_fer_log_to_sqlite({"emotion": "sad", "confidence": 0.89})
    logger.record_model_metrics({"latency": 0.045, "stt_confidence": 0.98})
    logger.record_test_report({"wer": 0.05, "cer": 0.02, "mos": 4.5, "quality_score": 92.0})
    
    report = logger.generate_test_report()
    assert report["status"] == "success"
    assert report["fer_logs_count"] == 1
    assert report["kpi_benchmarks_count"] == 1  # From record_model_metrics' fallback insert
    assert report["avg_wer"] == 0.05
    assert report["avg_cer"] == 0.02
    assert report["avg_mos"] == 4.5
    assert report["avg_quality_score"] == 92.0
