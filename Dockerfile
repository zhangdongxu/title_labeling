FROM ubuntu:16.04

RUN \
  apt-get update && \
  apt-get install -y python python-pip && \
  rm -rf /var/lib/apt/lists/* &&\
  apt-get clean -yq

# install grpcio
RUN pip install grpcio

ADD . /home/cpp/findMovieTitle
WORKDIR /home/cpp/findMovieTitle

EXPOSE 5011

CMD ["python", "movie_server.py", "--model", "model.p"]
