FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV REPOSITORY=HumanPoseEstimationComponent
ENV BUILD_USER=travis
ENV BUILD_HOME="/home/${BUILD_USER}"
ENV BUILD_DIR="${BUILD_HOME}/${REPOSITORY}"
ENV PATH="${BUILD_HOME}/.local/bin:${PATH}"
ENV VOLUME_DIR="${BUILD_HOME}/Volume"
ARG DEBIAN_FRONTEND=noninteractive

# set locale
ENV LANG en_GB.UTF-8
ENV LANGUAGE en_GB:en
ENV LC_ALL C

# set python encoding
ENV PYTHONIOENCODING=UTF-8

# substitute sh with bash
# install minimum requirements
# generate locale
# make non-interactive
RUN rm /bin/sh \
    && ln -s /bin/bash /bin/sh \
    && apt-get -qq update \
    && apt-get install -y apt-utils \
    && apt-get install -y -qq software-properties-common gpgv2 wget git nano lsb-release \
    && if [ "$(lsb_release -sc)" = "trusty" ]; then \
        apt-get install python-software-properties; \
        add-apt-repository -y ppa:tianon/gosu; \
        apt-get update -qq; \
       fi \
    && add-apt-repository -y ppa:tianon/gosu \
    && apt-get -qq update \
    && apt-get install -y git python3 python sudo locales curl gosu python-dev python3-dev \
    && echo 'debconf debconf/frontend select Noninteractive' \
    && DEBIAN_FRONTEND=noninteractive apt-get install --force-yes -y keyboard-configuration \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_GB.UTF-8

# create user
RUN useradd --create-home --shell /bin/bash ${BUILD_USER} \
    && echo "${BUILD_USER}:${BUILD_USER}" | chpasswd \
    && usermod -aG sudo ${BUILD_USER} \
    && echo '%sudo     ALL=NOPASSWD: ALL' > /etc/sudoers.d/sudo-nopasswd \
    && chmod 0440 /etc/sudoers.d/sudo-nopasswd


RUN apt-get update -qq \
    && apt-get install -y libcupti-dev libblas-dev liblapack-dev gfortran python-tk \
       libusb-1.0-0-dev libudev-dev openjdk-8-jdk freeglut3-dev doxygen graphviz python3-tk swig


COPY --chown=1000:1000 . ${BUILD_DIR}
WORKDIR ${BUILD_DIR}
USER ${BUILD_USER}

RUN sudo curl https://bootstrap.pypa.io/get-pip.py  -o get-pip.py \
    && python2 get-pip.py --user && python3 get-pip.py --user

RUN pip3 install --user tensorflow-gpu cython scipy sklearn pandas \
    scikit-image opencv-python protobuf

RUN cd Dependencies/OpenNI2 && make && cd Packaging && python ReleaseVersion.py x64

RUN cd Dependencies/tf-pose-estimation && pip3 install --user -r requirements.txt \
    && cd tf_pose/pafprocess && swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace


#RUN sh ./setup.sh


RUN sudo rm -rf /var/lib/apt/lists/*

VOLUME ${VOLUME_DIR}
#ENTRYPOINT ["bash"]
CMD ["bash"]
