```bash
python -m venv nextvis-env
cd nextvis-env
.\nextvis-env\Scripts\activate
pip install -r requirements.txt
python CV_matching.py path/to/cv/folder path/to/job/folder --output path/to/results --api_type openai --api_key your_api_key
python CV_Matching_Local.py path/to/cv/folder path/to/job/folder --output path/to/results
```
