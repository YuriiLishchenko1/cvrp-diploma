import json
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd


def save_run(result: dict, folder: str = 'runs') -> str:
    """
    Зберігає один run як json файл у папці runs/
    Повертає унікальний run_id (timestamp+uuid).
    """
    Path(folder).mkdir(exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + uuid.uuid4().hex[:6]
    p = Path(folder) / f"run_{run_id}.json"
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return run_id


def load_runs(files) -> pd.DataFrame:
    """
    Завантажує багато json-ів (UploadedFile або pathlib.Path) у DataFrame
    """
    records = []
    for f in files:
        if hasattr(f, 'read'):
            data = json.load(f)
        else:
            data = json.loads(Path(f).read_text(encoding='utf-8'))
        rec = {
            'run_id': data.get('file', '') + '_' + data.get('algo',''),
            'algo': data.get('algo',''),
            'distance': data.get('distance', 0),
            'vehicles': data.get('vehicles', 0),
            'time_sec': data.get('time_sec', 0)
        }
        records.append(rec)
    df = pd.DataFrame(records)
    return df
