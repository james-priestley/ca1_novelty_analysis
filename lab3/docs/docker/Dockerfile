FROM debian:latest

#  $ docker build . -t continuumio/anaconda:latest -t continuumio/anaconda:5.3.0 -t continuumio/anaconda2:latest -t continuumio/anaconda2:5.3.0
#  $ docker run --rm -it continuumio/anaconda2:latest /bin/bash
#  $ docker push continuumio/anaconda:latest
#  $ docker push continuumio/anaconda:5.3.0
#  $ docker push continuumio/anaconda2:latest
#  $ docker push continuumio/anaconda2:5.3.0

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

COPY ./dependencies.yml /dependencies.yml
COPY ./config /root/.ssh/config
COPY ./gitlab_key /root/.ssh/gitlab_key
COPY ./gitlab_key.pub /root/.ssh/gitlab_key.pub

RUN chmod a-rwx /root/.ssh/gitlab_key
RUN chmod u+r /root/.ssh/gitlab_key

RUN apt-get update
RUN apt-get -y install gcc
RUN conda install --yes ipykernel nb_conda_kernels
RUN conda env update --file /dependencies.yml
RUN conda run -n lab3 pip install suite2p # needs to be installed separately for some reason

RUN mkdir -p /root/code
RUN git clone git@gitlab.com:losonczylab/lab3.git /root/code/lab3
RUN git clone git@gitlab.com:losonczylab/sima.git /root/code/sima

RUN cd /root/code/lab3 && \
    conda run -n lab3 pip install -e . && \
	cd /root/code/sima && \
	conda run -n lab3 pip install -e .
	
RUN cd /root/code/lab3 && \
	conda run -n lab3 python -m ipykernel install --user --name lab3 --display-name "Python [conda env:lab3]"