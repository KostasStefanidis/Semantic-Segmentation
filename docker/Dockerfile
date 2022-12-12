FROM tensorflow/tensorflow:latest-gpu
#tensorflow/tensorflow:2.10.1-gpu

ARG USERNAME=kstef

RUN apt-get install sudo -y

RUN useradd -m -G sudo -s /bin/bash $USERNAME

RUN mkdir -p /home/$USERNAME/.vscode-server/extensions \
        /home/$USERNAME/.vscode-server-insiders/extensions \
    && chown -R $USERNAME \
        /home/$USERNAME/.vscode-server \
        /home/$USERNAME/.vscode-server-insiders

RUN apt-get install git -y \
&& apt-get install zip -y \
&& apt-get install unzip -y \
&& apt-get install wget -y

RUN pip install --upgrade pip \
&& pip install ipython \
&& pip install ipykernel \
&& pip install tensorflow-addons \
&& pip install pandas \
&& pip install pydot \
&& apt-get install graphviz -y

RUN git config --global user.name $USERNAME@tf-container \
&& git config --global user.email kstefanidis48@gmail.com