FROM ros:noetic

RUN apt-get update -y && apt-get install -y ros-noetic-compressed-image-transport
RUN apt-get update -y && apt-get install -y python3-pip
#RUN pip install tensorflow keras torch
RUN pip install torch torchvision
#--extra-index-url https://download.pytorch.org/whl/cpu
# python-dateutil==2.8.1 fixes a warning/requirment for pandas
RUN pip install matplotlib pandas python-dateutil==2.8.1
RUN rm -rf /root/.cache/pip/
WORKDIR /root/projects/
RUN echo 'catkin_make install && source /root/projects/devel/setup.bash' >> /root/.bashrc
#WORKDIR /root/projects/src
