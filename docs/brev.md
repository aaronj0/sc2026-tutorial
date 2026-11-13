# Running the tutorial on NVIDIA Brev

[NVIDIA Brev](https://brev.nvidia.com) runs the tutorial on a cloud GPU and opens JupyterLab in your browser.
You need a laptop and a browser.

## Join the organization

- Open the invite link from your instructor.
- Log in, or sign up with your email: choose a password, enter the 6-digit code from your email, then create
  the NVIDIA Cloud Account with your name.
- The organization name at the top right must read `SC26-Python-HPC`. If it does not, click it and switch.

## Start the tutorial

- Click "Launchables" in the top bar, then "My Launchables".
- Click "View Launchable" on the tutorial, then "Deploy Launchable", then "Go to Instance Page".
- Wait for both status indicators to turn green: "Running" after 1 to 3 minutes, "Built" after another
  4 to 10 minutes.
- Click "Open Notebook" when it flashes green. If it does not within a minute, refresh the page.
- JupyterLab opens on the tutorial outline. Its table links every notebook of the day.

## Work in JupyterLab

- Exercises have `TODO` cells. The answers are in the `solutions/` folder next to each notebook.
- Run one notebook at a time. Before opening the next, shut down the current kernel
  ("Kernel > Shut Down Kernel") to free the GPU memory.
- Nsight Systems and Nsight Compute open from the "nsys" and "ncu" tiles on the JupyterLab launcher.
- Your edits live on the instance. To keep a notebook, right-click it and choose "Download".

## Afterwards

Stop or delete the instance from its page in the Brev console.
