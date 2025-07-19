create a virtual env:
```
python -m venv env
```
activate the virtual environment (for Windows):
```
./env/Scripts/Activate
```
install all required packages:
```
pip install -r requirements.txt
```
run docker:
```
docker pull qdrant/qdrant                                        docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
run the streamlit application:
```
streamlit run frontend.py
```