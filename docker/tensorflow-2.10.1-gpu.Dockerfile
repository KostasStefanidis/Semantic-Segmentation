FROM tensorflow/tensorflow:2.10.1-gpu

ARG USERNAME=kstef

RUN apt-get install sudo -y

RUN useradd -m -G sudo -s /bin/bash $USERNAME

RUN mkdir -p /home/$USERNAME/.vscode-server/extensions \
        /home/$USERNAME/.vscode-server-insiders/extensions \
    && chown -R $USERNAME \
        /home/$USERNAME/.vscode-server \
        /home/$USERNAME/.vscode-server-insiders

# install usefull linux packages
RUN apt-get install git -y \
&& apt-get install zip -y \
&& apt-get install unzip -y \
&& apt-get install wget -y \
&& apt install screen -y

# Configure git
RUN git config --global user.name $USERNAME \
&& git config --global user.email kstefanidis48@gmail.com

# install usefull python packages
RUN pip install --upgrade pip \
&& pip install ipython \
&& pip install ipykernel \ 
&& pip install tensorflow-addons \
&& pip install scipy \
&& pip install pandas \
&& pip install matplotlib \
&& pip install -U scikit-learn

RUN pip install tf-models-official