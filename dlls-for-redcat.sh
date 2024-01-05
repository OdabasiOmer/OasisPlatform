#!/bin/bash

sudo docker cp /usr/lib/x86_64-linux-gnu/libgfortran.so.4 oasisplatform_worker_1:lib/x86_64-linux-gnu
sudo docker cp /usr/lib/x86_64-linux-gnu/libgfortran.so.4.0.0 oasisplatform_worker_1:lib/x86_64-linux-gnu

sudo docker cp /usr/lib/x86_64-linux-gnu/libgomp.so.1 oasisplatform_worker_1:lib/x86_64-linux-gnu
sudo docker cp /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0 oasisplatform_worker_1:lib/x86_64-linux-gnu

sudo docker cp /usr/lib/x86_64-linux-gnu/libquadmath.so.0 oasisplatform_worker_1:lib/x86_64-linux-gnu
sudo docker cp /usr/lib/x86_64-linux-gnu/libquadmath.so.0.0.0 oasisplatform_worker_1:lib/x86_64-linux-gnu

sudo docker cp /usr/lib/x86_64-linux-gnu/libf2c.so.0 oasisplatform_worker_1:lib/x86_64-linux-gnu
sudo docker cp /usr/lib/x86_64-linux-gnu/libf2c.so.0.11 oasisplatform_worker_1:lib/x86_64-linux-gnu
