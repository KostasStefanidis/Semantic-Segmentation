FROM tensorflow/tensorflow:latest
#tensorflow/tensorflow:2.10.1-gpu

RUN apt install sudo -y

RUN useradd -m -G sudo -s /bin/bash kstef

RUN pip install --upgrade pip \
&& pip install ipython \
&& pip install ipykernel \
&& pip install pydot \
&& apt install graphviz -y

RUN apt install git -y \
&& apt install zip -y \
&& apt install unzip -y \
&& apt install curl -y \
&& apt install wget -y

RUN git config --global user.name kstef@tf-container \
&& git config --global user.email kstefanidis48@gmail.com