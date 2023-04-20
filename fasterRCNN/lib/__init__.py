# sudo docker run -i -t -v `pwd`:/io quay.io/pypa/manylinux_2_28_x86_64 /bin/bash
# yum install python38-devel
# cd io/fasterRCNN/lib
# python3 -m pip install --upgrade pip
# python3 -m pip install torch==1.10.2
# python3 -m pip install -r pip-requirements.txt
# python3 setup.py build develop

# python3 setup.py bdist_wheel
# OR
# python3 setup.py sdist
# python -m pip wheel .

# export LD_LIBRARY_PATH=/usr/local/lib64/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/io/fasterRCNN/lib/model:$LD_LIBRARY_PATH

# /io/fasterRCNN/lib/build/temp.linux-x86_64-3.8/io/fasterRCNN/lib/model/csrc/cpu

# rm -rf build && rm -rf faster_rcnn.egg-info && rm -rf dist && rm -f model/_C.cpython-38-x86_64-linux-gnu.so

# python3 -m zipfile --list dist/faster_rcnn-0.1-cp38-cp38-linux_x86_64.whl | grep model

# sudo docker exec -ti cb9b86f8ac0a sh -c "/bin/bash -c 'cd /home/model-server/STTran && rm -r build && rm -r dist && rm -r sttran.egg-info && python setup.py build develop && python setup.py bdist_wheel && python -m pip install --force-reinstall dist/sttran-0.1-cp38-cp38-linux_x86_64.whl'"
