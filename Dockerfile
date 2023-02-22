FROM python
WORKDIR /python2
COPY . /python2
EXPOSE 8513
RUN pip install -r requirements.txt
CMD streamlit run server.py
