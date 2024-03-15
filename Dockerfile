FROM python:3.7-slim-bullseye
WORKDIR ./ImageAnalysisService
ADD . .
RUN pip config --user set global.progress_bar off
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y poppler-utils
# 将容器暴露在指定的端口上
EXPOSE 18220
CMD ["python", "./service_run.py"]