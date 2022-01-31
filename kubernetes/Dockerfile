FROM pytorch/pytorch
# workdir is /workspace/
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY util util/
COPY model model/
COPY run.py .
COPY tmp.json .

RUN chown -R 1147:1003 .