FROM registry.access.redhat.com/ubi9/python-311

USER root
WORKDIR /app/

# install and activate env
COPY requirements.txt requirements.txt
RUN pip3 install uv
RUN uv pip install -r requirements.txt

COPY models/ models/
COPY feature_repo/ feature_repo/
COPY service-ca.crt service-ca.crt
# give premisssions and 
RUN chmod -R 777 . && ls -la