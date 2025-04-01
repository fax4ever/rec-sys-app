FROM registry.access.redhat.com/ubi9/python-311

USER root
WORKDIR /app/

# install and activate env
COPY pyproject.toml pyproject.toml
RUN pip install uv -y
RUN uv sync
ENV VIRTUAL_ENV=.venv
ENV PATH=".venv/bin:$PATH"

COPY data/ data/
COPY feature_store.yaml feature_store.yaml
# give premisssions and 
RUN chmod -R 777 . && ls -la