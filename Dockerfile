FROM ubuntu:16.04

RUN \
  apt-get update && \
  apt-get install -y python3 python3-pip && \
  rm -rf /var/lib/apt/lists/* &&\
  apt-get clean -yq

# install grpcio
RUN pip3 install grpcio

ADD . /home/cpp/title_labeling
WORKDIR /home/cpp/title_labeling

ENV LANG=C.UTF-8

EXPOSE 5011

CMD ["sh", "movie_server.sh"]
