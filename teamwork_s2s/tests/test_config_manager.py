import os
import json
import pytest
from teamwork_s2s.src.config.config_manager import ConfigManager

def test_config_manager_defaults():
    cm = ConfigManager()
    assert cm.get("vad_threshold") == 0.01
    assert cm.get("stt_model") == "mock"
    assert cm.get("fer_enabled") is True

def test_config_manager_initial_config():
    cm = ConfigManager(initial_config={"vad_threshold": 0.05})
    assert cm.get("vad_threshold") == 0.05
    assert cm.get("stt_model") == "mock"  # retains default

def test_config_manager_update():
    cm = ConfigManager()
    res = cm.update_config({"vad_threshold": 0.02})
    assert res is True
    assert cm.get("vad_threshold") == 0.02

def test_config_file_loading_and_hot_reload(tmp_path):
    config_file = tmp_path / "config.json"
    initial_data = {"vad_threshold": 0.02, "stt_model": "whisper"}
    
    with open(config_file, "w") as f:
        json.dump(initial_data, f)
        
    cm = ConfigManager(config_path=str(config_file))
    assert cm.get("vad_threshold") == 0.02
    assert cm.get("stt_model") == "whisper"

    # Hot-reload scenario
    new_data = {"vad_threshold": 0.08, "stt_model": "whisper"}
    with open(config_file, "w") as f:
        json.dump(new_data, f)
        
    cm.reload_config()
    assert cm.get("vad_threshold") == 0.08

def test_config_manager_validation():
    cm = ConfigManager()
    # Type validation failures should trigger rollback
    res = cm.update_config({"vad_threshold": "invalid_string_type"})
    assert res is False
    assert cm.get("vad_threshold") == 0.01
