# up4-GUI
A GUI for up4.

## Installation

Please install the requirements in requirements.txt.
You will also need to install up4. Features of the dispersion plot are currently being developed through a special version of up4, to be uploaded. 

## Interactive plotting on Bluebear (Linux/Mac/WSL)
This allows you to run the up4 GUI without local installation. Drop the `up4_gui.py` file into your Bluebear directory. Then, enter the following into your Linux/Mac/WSL terminal window:

    ssh -X [user]@bluebear.bham.ac.uk
 
 You will be asked to authenticate and confirm connection. Then enter: 

    module load slurm-interactive
    fisbatch_screen --nodes=1-1 --ntasks=16 --time=2:0:0 

Change time & number of cores as required. You will then be directed to an interactive session node.
Enter the following:

    module load bear-apps/2022b
    module load Python/3.10.8-GCCcore-12.2.0
    module load HDF5
    module load Tkinter/3.10.8-GCCcore-12.2.0
    module load Rust/1.73.0-GCCcore-12.2.0
    module load SciPy-bundle/2023.02-gfbf-2022b
    module load plotly.py/5.13.1-GCCcore-12.2.0
    pip install git+https://github.com/uob-positron-imaging-centre/up4.git
    pip install dash
 

   Change directory to the folder containing `up4_gui.py`, then run:

    python3 up4_gui.py

A dialog box will open for you to select your `.hdf5` file, which you should do. Take a note of your node pg code ("`pg****u**a`"), and the port number the program is being hosted on.

In a second terminal window, enter:

    ssh -L [port number]:localhost:[port number] -J [user]@bluebear.bham.ac.uk -v [user]@bear-[pg code]
    
You will need to enter your BEAR password a couple of times to authenticate yourself, then as soon as you're logged in you should be able to run the GUI in your browser using the link provided by your first terminal window.

