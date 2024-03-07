FROM python:3.9
WORKDIR ./ImageAnalysisService
ADD . .
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y poppler-utils
# 将容器暴露在指定的端口上
EXPOSE 18220
CMD ["python", "./service_run.py"]