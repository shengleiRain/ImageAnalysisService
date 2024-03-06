FROM python:3.9
WORKDIR ./ImageAnalysisService
ADD . .
RUN pip install -r requirements.txt
# 将容器暴露在指定的端口上
EXPOSE 18220
CMD ["python", "./service_run.py"]