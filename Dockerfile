FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /workspace/GeneSPT

COPY requirements.txt /tmp/genespt-requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/genespt-requirements.txt \
    && rm -f /tmp/genespt-requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/GeneSPT/main:/workspace/GeneSPT/scripts
ENV MPLBACKEND=Agg
