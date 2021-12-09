FROM python:3.8
COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY ./front_end /front_end
WORKDIR "/front_end"
EXPOSE 8050
CMD ["python", "application.py"]



