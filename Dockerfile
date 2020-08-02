FROM nvidia/cuda:10.2-base

# set up environment
RUN apt-get update --fix-missing && \ 
	apt-get install -q -y --no-install-recommends \
	curl \
	build-essential	\
	sudo \
	vim \
	git

# install conda
RUN cd /tmp && \
	curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && \
	sha256sum Anaconda3-2020.02-Linux-x86_64.sh && \
	bash Anaconda3-2020.02-Linux-x86_64.sh -b -p ~/anaconda && \
	rm Anaconda3-2020.02-Linux-x86_64.sh && \
	echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc

# prepare conda environment
ENV CONDA_ENV_NAME howl

WORKDIR /app
COPY environment.yml /app
RUN ~/anaconda/bin/conda init bash && \
        ~/anaconda/bin/conda env create -n ${CONDA_ENV_NAME} -f environment.yml

# prepare data folder
RUN mkdir /data
