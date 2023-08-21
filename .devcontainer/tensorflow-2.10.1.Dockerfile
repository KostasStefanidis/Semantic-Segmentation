FROM tensorflow/tensorflow:2.10.1-gpu

ARG OS_USER_ID
ARG OS_GROUP_ID
ARG USERNAME
ARG GIT_USER_EMAIL

RUN groupadd --gid ${OS_GROUP_ID} ${USERNAME} \
&& useradd --uid ${OS_USER_ID} --gid ${OS_GROUP_ID} --shell /bin/bash --create-home --no-user-group ${USERNAME} \
&& chown ${OS_USER_ID}:${OS_GROUP_ID} /mnt/

# install usefull linux packages
RUN apt-get update \
&& apt-get install git -y \
&& apt-get install zip -y \
&& apt-get install unzip -y \
&& apt-get install wget -y \
&& apt-get install curl -y \
&& apt-get install screen -y

USER ${OS_USER_ID}

# install usefull python packages
RUN pip install --no-cache-dir --upgrade pip \
&& pip install --no-cache-dir ipython==8.10.0 \
&& pip install --no-cache-dir ipykernel==6.22.0 \
&& pip install --no-cache-dir pandas==2.0.0 \
&& pip install --no-cache-dir matplotlib==3.7.1 \
&& pip install --no-cache-dir -U scikit-learn==1.2.2 \
&& pip install --no-cache-dir PyYAML==6.0

# configure git
RUN git config --global user.name ${USERNAME} \
&& git config --global user.email ${GIT_USER_EMAIL}