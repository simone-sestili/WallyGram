CONTAINER_NAME=wallygram_1

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

docker run -it --name ${CONTAINER_NAME} wallygram:1.0