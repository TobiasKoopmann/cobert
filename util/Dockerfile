FROM pytorch/pytorch
# workdir is /workspace/
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY util/create_temporal_dataset.py create_temporal_dataset.py
COPY util/PaperEmbeddings.py util/PaperEmbeddings.py

RUN chown -R 1147:1003 .