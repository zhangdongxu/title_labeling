FROM ubuntu:16.04

# install bazel
# last line of install packages are Bazel requirements
RUN \
  apt-get update && \
  apt-get install -y git man gcc vim-nox cscope exuberant-ctags silversearcher-ag \
                    screen wget curl libcurl4-openssl-dev xz-utils ncdu pax-utils \
                    pkg-config zlib1g-dev python unzip zip g++ bash-completion \
                    make bison flex &&\
  rm -rf /var/lib/apt/lists/* &&\
  apt-get clean -yq
  

# donwload java 8 directly and extract
RUN wget -qO-  --no-check-certificate --no-cookies --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/8u45-b14/jdk-8u45-linux-x64.tar.gz|tar xvz
ENV JAVA_HOME /jdk1.8.0_45
ENV PATH $JAVA_HOME/bin:$PATH

# fix the bazel building sandbox issue by disable sandboxing
RUN echo "startup --batch" >>/tmp/bazelrc &&\
    echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >>/tmp/bazelrc

# build bazel from src (tag: 1/25)
RUN git clone https://github.com/google/bazel.git /bazel && cd /bazel && git checkout 0.3.1 &&\
    BAZELRC=/tmp/bazelrc /bazel/compile.sh
    
ENV PATH $PATH:/bazel/output/

# config the bazel command complete
RUN cd /bazel && \
    bazel --bazelrc=/tmp/bazelrc build //scripts:bazel-complete.bash && \
    cp bazel-bin/scripts/bazel-complete.bash /etc/bash_completion.d

# create bazelrc
RUN cd ~ &&\
    echo "build:arm --crosstool_top=//tools:toolchain --cpu=armeabi-v7a" >>.bazelrc &&\
    echo "build:i686 --crosstool_top=//tools:toolchain --cpu=i686" >>.bazelrc &&\
    echo "build:x86_64 --crosstool_top=//tools:toolchain --cpu=x86_64" >>.bazelrc

# get the bazel layer
RUN cd ~ &&\
    echo "rm -fr blayer" >>bazel_layer.sh &&\
    echo "git clone https://github.com/chin33z/bazel_buildfiles.git blayer" >>bazel_layer.sh &&\
    chmod u+x bazel_layer.sh

# 1. git clone the vim setting
# 2. git clone the env setting
RUN \
  rm -fr ~/.vim ~/.vimrc &&\
  cd ~ && git clone https://github.com/chin33z/dotvim.git ~/.vim &&\
  ln -s ~/.vim/vimrc ~/.vimrc &&\
  cd ~ && git clone https://github.com/chin33z/dotfiles.git ~/dotfiles &&\
  cd dotfiles && ./link.sh

ENV HOME /root
WORKDIR /root

# install bazel finish


RUN apt-get update \
    && apt-get install -y apt-utils
RUN apt-get install -y autoconf automake libtool curl make unzip wget git

# install protobuf v3.0.0-beta-2
WORKDIR /home/cpp/
RUN wget https://github.com/google/protobuf/releases/download/v3.0.0-beta-2/protobuf-cpp-3.0.0-beta-2.zip \
    && unzip protobuf-cpp-3.0.0-beta-2.zip
WORKDIR /home/cpp/protobuf-3.0.0-beta-2
RUN ./configure \
    && make \
    && make check \
    && make install \
    && ldconfig

# install grpc
WORKDIR /home/cpp/
RUN git clone https://github.com/grpc/grpc.git
WORKDIR /home/cpp/grpc
RUN git fetch origin release-0_14   \
    && git checkout -b release-0_14 origin/release-0_14 \
    && git submodule update --init  \
    && make \
    && make install

# install glog & gflags
RUN apt-get update \
    && apt-get install -y libgoogle-glog-dev

RUN apt-get install -y python-pip 
RUN pip install numpy && pip install grpcio

ADD . /home/cpp/title_labeling
WORKDIR /home/cpp/title_labeling

EXPOSE 5011

CMD /bin/bash
