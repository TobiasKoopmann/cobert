FROM pytorch/pytorch
# workdir is /workspace/
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN chown -R 1147:1003 .