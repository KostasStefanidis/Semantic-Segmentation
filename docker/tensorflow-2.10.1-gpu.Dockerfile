FROM tensorflow/tensorflow:2.10.1-gpu

ARG USERNAME
ARG USER_ID
ARG GROUP_ID

#RUN apt install sudo=1.8.31-1ubuntu1 -y
RUN groupadd --gid ${GROUP_ID} $USERNAME
RUN useradd --uid ${USER_ID} --gid ${GROUP_ID} --shell /bin/bash --create-home --no-user-group ${USERNAME}
RUN chown ${USER_ID}:${GROUP_ID} /mnt/

# install usefull linux packages
RUN apt update \
&& apt install git=1:2.25.1-1ubuntu3.10 -y \
&& apt install zip=3.0-11build1 -y \
&& apt install unzip -y \
&& apt install wget=1.20.3-1ubuntu2 -y \
&& apt install curl=7.68.0-1ubuntu2.18 -y \
&& apt install screen=4.8.0-1ubuntu0.1 -y

USER ${USER_ID}

# install usefull python packages
RUN pip install --upgrade pip \
&& pip install ipython \
&& pip install ipykernel \
&& pip install tensorflow-addons \
&& pip install scipy \
&& pip install pandas \
&& pip install matplotlib \
&& pip install -U scikit-learn \
&& pip install PyYAML

RUN git config --global user.name ${USERNAME} \
&& git config --global user.email kstefanidis48@gmail.com