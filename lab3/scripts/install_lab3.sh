#!/bin/bash

SIMA_URL=git@gitlab.com:losonczylab/sima.git
#CONDA_URL=https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
CONDA_URL=https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
INSTALL_DIR=~/code

progress=$RANDOM$RANDOM$RANDOM$RANDOM$RANDOM$RANDOM$RANDOM$RANDOM$RANDOM
touch $progress
printf "wait" > $progress
sp='/-\|'
sc=0

function ProgressBar {
    while true
    do
        if [ "${1}" != "wait" ]; then
            PROGRESS=$(cat ${1} 2>/dev/null) 2>/dev/null
            let _progress=($PROGRESS*100/100*100)/100 2>/dev/null
            let _done=(${_progress}*4)/10 2>/dev/null
            let _left=40-$_done 2>/dev/null
            _fill=$(printf "%${_done}s") 2>/dev/null
            _empty=$(printf "%${_left}s") 2>/dev/null

            printf "\r[${_fill// /#}${_empty// /-}] ${_progress}%%  \b${sp:sc++:1}" 2>/dev/null
            ((sc==${#sp})) && sc=0 2>/dev/null
        fi
        sleep 0.1
    done

}

# Will ignore the error message if code directory already exists.
mkdir $INSTALL_DIR 2>/dev/null

# Create log files. Standard and error output will be redirected to these.
DATE=$(date +%F:%T | tr ":" "_" | tr "-" "_")

LOG="$INSTALL_DIR/installation_$DATE.log"
touch $LOG

printf "\n1) Please make sure you have access to the sima repo ($SIMA_URL) before attempting to install and\n"
printf "\n2) Please make sure your gitlab ssh does not require a passphrase\n"
printf "\nInstalling lab3. This will take some time.\n\n"
start=`date +%s`

main() {
    # remove this directory only if they are empty
    rm -d $INSTALL_DIR/sima 1>>"$LOG" 2>>"$LOG"

    # check if $INSTALL_DIR/sima still exists
    SIMA_EXISTS="$(ls -A $INSTALL_DIR/sima 2>/dev/null)"

    if [ ! -z "$SIMA_EXISTS" ]; then
            printf "\nA non-empty sima folder already exists in $INSTALL_DIR. Aborting installation.\n"
            return
    fi

    # install gcc -- will be needed to build sima!
    sudo apt-get update 1>>"$LOG" 2>>"$LOG"
    sudo apt-get --assume-yes install gcc 1>>"$LOG" 2>>"$LOG"

    printf 5 > ${1}

    wget $CONDA_URL -O ./anaconda_install.sh 1>>"$LOG" 2>/dev/null
    printf 10 > ${1}
    chmod +x ./anaconda_install.sh 1>>"$LOG" 2>>"$LOG"
    bash ./anaconda_install.sh -b -fp ~/anaconda 1>>"$LOG" 2>>"$LOG"
    eval "$(~/anaconda/bin/conda shell.bash hook)" 1>>"$LOG" 2>>"$LOG"
    printf 25 > ${1}
    
    cd $INSTALL_DIR/lab3/scripts 1>>"$LOG" 2>>"$LOG"

    #conda activate base 1>>"$LOG" 2>>"$LOG"
    #conda install --yes ipykernel nb_conda_kernels 1>>"$LOG" 2>>"$LOG"
    #conda deactivate 1>>"$LOG" 2>>"$LOG"
    #printf 40 > ${1}
    
    conda env update --file dependencies.yml 1>>"$LOG" 2>>"$LOG"
    conda activate lab3 1>>"$LOG" 2>>"$LOG"
    pip install -e .. 1>>"$LOG" 2>>"$LOG"
    printf 80 > ${1}

    cd $INSTALL_DIR 1>>"$LOG" 2>>"$LOG"
    git clone $SIMA_URL 1>>"$LOG" 2>>"$LOG"
    pip install -e sima 1>>"$LOG" 2>>"$LOG"
    printf 100 > ${1}

    echo "export PATH="$HOME/anaconda/condabin:$PATH"" >> ~/.bash_profile
    echo "source ~/.bashrc" >> ~/.bash_profile
}

ProgressBar $progress &
MYSELF=$!

main $progress
kill $MYSELF >/dev/null 2>&1
rm $progress >/dev/null 2>&1

conda run -n lab3 pip install suite2p==0.7.5 # for some reason putting suite in pip section of dependencies.yml doesn't work

printf "\r[########################################] 100%%    \n"

end=`date +%s`
printf "\nInstalled lab3 in $(((end-start)/60)) minutes.\n"

printf "\n\n************************************************************\n"
printf "*** ! Log out and log back in to complete installation ! ***\n"
printf "************************************************************\n\n"
