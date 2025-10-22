.PHONY: api app ingest fmt onnx


api:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000


app:
streamlit run app/pages/home.py


ingest:
python scripts/ingest_docs.py --dir docs


fmt:
python -m pip install -q ruff black && ruff check . --fix && black .


onnx:
python scripts/export_onnx.py