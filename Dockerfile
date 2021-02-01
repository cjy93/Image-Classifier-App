# Add a line here to specify the docker image to inherit from.
FROM registry.aisingapore.net/aiap/polyaxon/pytorch-tf2-cpu:latest

ARG WORK_DIR="/home/polyaxon"
ARG USER="polyaxon"

WORKDIR $WORK_DIR

# Add lines here to copy over your src folder and 
# any other files you need in the image (like the saved model).
# COPY file
COPY src src/
COPY conda.yml README.md tensorfood.h5 ./

# Add a line here to update the base conda environment using the conda.yml. 
RUN conda env update -f conda.yml -n base && \
    rm conda.yml

# DO NOT remove the following line - it is required for deployment on Tekong
RUN chown -R 1000450000:0 $WORK_DIR

USER $USER

EXPOSE 8000
# behind 8000

# Add a line here to run your app
CMD [ "python" , "-m", "src.app"]