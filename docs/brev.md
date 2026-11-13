# Running the tutorial on NVIDIA Brev

[NVIDIA Brev](https://brev.nvidia.com) runs the tutorial's container stack on a cloud GPU and opens
JupyterLab in the browser. Attendees need a laptop and a browser. A deployment starts
[brev/docker-compose.yml](../brev/docker-compose.yml): JupyterLab from `ghcr.io/pyhpc/sc2026-tutorial:latest`
on port 8888, opening on the tutorial outline, with the Nsight Systems and Nsight Compute GUIs on ports 8080
and 8081. Nothing is built on the instance.

## Attendees: deploy the Launchable

- Go to the team invite URL from your instructor and log in or sign up: email, password or a federated
  sign-in, the 6-digit code from your email, then "Create NVIDIA Cloud Account" with your name.
- Check the organization name at the top right: it must read `SC26-Python-HPC`. Click it to switch.
- Click "Launchables" on the navigation bar, then "My Launchables", then "View Launchable" on the
  tutorial's Launchable and "Deploy Launchable". With a Launchable URL from your instructor, open it and
  click "Deploy Launchable".
- Click "Go to Instance Page". Two status indicators appear. "Deploying" turns into a green "Running"
  after 1 to 3 minutes; the second one turns into a green "Built" after 4 to 10 more minutes while the
  instance pulls the images.
- Click "Open Notebook" once it flashes green. If it does not within a minute of both indicators being
  green, refresh the page.
- JupyterLab opens on the outline, `notebooks/syllabi/sc26__8_hours.ipynb`, whose table links every
  notebook of the day. Exercises carry `TODO` cells; the `solutions/` notebooks hold the answers.
- Run one notebook at a time: shut down its kernel ("Kernel > Shut Down Kernel") before opening the
  next, so that it releases the GPU memory. The mini-app notebooks run in their numbered order.
- Nsight Systems and Nsight Compute are the "nsys" and "ncu" tiles on the JupyterLab launcher.
- Your edits live on the instance. Download what you want to keep (right-click, "Download") before
  the instance is deleted. Afterwards, stop or delete the instance from its page.

## Organizers: create the Launchable

Once, in the `SC26-Python-HPC` organization: "Launchables", "Create Launchable".

- Code: "I have code files in a git repository", URL `https://github.com/pyHPC/sc2026-tutorial`,
  "With container(s)".
- Container: "Docker Compose", "I have an existing docker-compose.yaml file", "Provide GitHub/Gitlab URL",
  `https://github.com/pyHPC/sc2026-tutorial/blob/main/brev/docker-compose.yml`, "Validate".
- Services: "No, I don't want Jupyter" (the compose file provides it). Under "Secure Links" delete the
  links for ports 3478 and 3479, name 8888 "jupyter", 8080 "nsys" and 8081 "ncu". Under "TCP/UDP Ports"
  add the rule "3478-3479" (the Nsight streamers' TURN ports).
- Compute: one NVIDIA L4 on a provider with Flexible Ports (GCP `g2-standard-8`), host driver with
  CUDA 13 support, 128 GB storage (the tutorial image is 10 to 13 GB uncompressed plus the two Nsight
  images and the repository volume).
- Review: name it and "Create Launchable". "View Deploy Page" gives the Launchable URL; the
  organization's settings give the team invite URL for the attendees.
- Deploy it yourself once and run the outline's first notebooks before sending the URL.
