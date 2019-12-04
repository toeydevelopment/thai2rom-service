FROM pytorch/pytorch

RUN mkdir /app
WORKDIR /app

ADD . .
RUN pip install numpy 
RUN pip install flask 
RUN pip install flask_cors 
RUN pip install -U --pre pythainlp
RUN pip install pandas
EXPOSE 5000

CMD [ "python","app.py" ]