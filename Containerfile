FROM registry.access.redhat.com/ubi9/python-311

USER root
WORKDIR /app/

# install and activate env
COPY requirements.txt requirements.txt
RUN pip3 install uv
RUN uv pip install -r requirements.txt
RUN dnf update -y && \
    dnf install -y wget && \
    wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq &&\
    chmod +x /usr/local/bin/yq

COPY models/ models/
COPY feature_repo/ feature_repo/
COPY entry_point.sh entry_point.sh
# give premisssions 
RUN chmod -R 777 . && ls -la

ENTRYPOINT ["/bin/sh", "-c", "/app/entry_point.sh"]