FROM pytorch/pytorch
# workdir is /workspace/
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY util util/
COPY model model/
COPY run.py .
COPY tmp.json .

# RUN apt-get update
# RUN apt-get install -y openssh-client

RUN chown -R 1147:1003 .