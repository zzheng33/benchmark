# Run on machines/testbeds with no root privs in ${MINIGAN_ROOT}
 
PIP=pip3
PYT=python3

#${PIP} install --upgrade pip --user



${PYT} -m venv minigan_env

source minigan_env/bin/activate

${PIP} install numpy 
${PIP} install wheel
${PIP} install torch
${PIP} install torchvision
${PIP} install horovod==0.18.2
${PIP} install tensorboard
${PIP} install matplotlib

deactivate
