============
Quickstart
============
Clone lab3
------------------

First clone the repository::

    cd ~/code
    git clone git@gitlab.com:losonczylab/lab3.git
    
Run the install script
------------------
Change directory into the `scripts` folder and make the install script executable
    cd lab3/scripts
    chmod +x ./install_lab3.sh

Run the install script
    ./install_lab3.sh

That's it! You're set up with lab3! 
	ipython
	>>> import lab3

To turn off activating conda at login, run
	conda config --set auto_activate_base false

To activate lab3 manually, simply
	conda activate lab3

