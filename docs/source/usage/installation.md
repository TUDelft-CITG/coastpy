# Installation Guide

Welcome to the installation guide for CoastPy! This document provides a
step-by-step guide to set up your coastal computing environment on a local machine. These
instructions are mostly written for Windows users, with no prior experience in Git.
Please adopt your usual workflows if that's not you.

## 1. Setting up Git

Git is a version control system that we use for managing the course materials. If you're
new to Git, we recommend you to start with [this
introduction](https://earth-env-data-science.github.io/lectures/environment/intro_to_git.html).

1. **Install Git software**:

   <details>
   <summary><strong>By GitHub Desktop</strong></summary>

   1. Follow the [GitHub Client documentation](https://desktop.github.com/) to
   install the GitHub client.

   2. GitHub client does not install the underlying git software on your machine. Follow [these
   instructions](https://learn.microsoft.com/en-us/devops/develop/git/install-and-set-up-git)
   to install git on your machine.
   </details>

   <details>

   <summary><strong>By command line</strong></summary>


   Follow [these instructions](https://github.com/git-guides/install-git) to install git using the
   command line.
   </details>

2. **Fork the repository**:

   Before you can start working on the project, you'll need to create a fork of the repository. This will give you a copy of the project in your GitHub account, allowing you to make changes without affecting the original project.

   <details>
   <summary><strong>By GitHub Desktop</strong></summary>

   1. Open GitHub Desktop > go to File > Clone Repository and paste this url: https://github.com/TUDelft-CITG/coastpy.git
   2. Choose where to clone the repository on your computer and click "Clone".
   </details>

   <details>
   <summary><strong>By command line</strong></summary>
   git clone https://github.com/TUDelft-CITG/coastpy.git
   </details>

Following these steps, the repository's files from GitHub are fetched to your machine.
You can verify this by navigating to the directory's path where you have cloned the
repository.


## 2. Installing Miniforge that comes with Mamba dependency solver

Miniforge is a lightweight installer for conda, specific to the conda-forge channel. We
recommend conda with mamba's solver to manage Python environments. If you're not familiar
with managing Python environments, please have a look at this
[introduction](https://earth-env-data-science.github.io/lectures/environment/python_environments.html?highlight=conda)
first. The software can be downloaded from the [Conda Forge
GitHub](https://github.com/conda-forge/miniforge#mambaforge) .

**Known issue**: Some users have their firewalls configured in such way that the
mambaforge installation is blocked. If you have trouble installing mambaforge, please make
sure to temporarily disable your firewall.

</details>

<details>

<summary><strong>Unix-like Systems (Mac and Linux)</strong></summary>

1. Open a terminal. On Mac, search for terminal or iterm in Spotlight. On linux, the
   hotkey to open a terminal is "cntrl + shift + t".
2. The commands to install the package manager are copied from their documentation ---
   double check to see if they are still the correct!
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
3. Accept the user agreements, and allow the installation script to edit your profile
   file because it ensures that the mamba command becomes available in your profile.

</details>

## 3. Creating the software environment

The notebooks require specific packages, which we have bundled in a coastal
environment.

<details>
<summary><strong>Windows Users</strong></summary>

1. On Windows, open a Miniforge Prompt by searching for "miniforge" in the task bar.
2. Change to the directory where you cloned the repository. If you installed the GitHub client using their default settings you run
   `cd%userprofile%\Documents\GitHub\coastpy`. By running `DIR` you can see a
   list of all files and directories. You can also see this in the file explorer by
   navigating to this directory.
3. The directory contains
   [environment.yml](https://github.com/floriscalkoen/coastpy/environment.yml),
   which is a file that describes the software dependencies. Now create the software
   environment by running the following command in the terminal/Miniforge prompt:

   ```bash
      mamba env create -f environment.yml
   ```

</details>

<details>
<summary><strong>Unix-like Systems (Mac and Linux)</strong></summary>

1. On Mac, search for terminal or iterm in Spotlight (command + space). On linux, the
   hotkey to open a terminal is "cntrl + shift + t".

2. You can navigate the terminal using `cd`, which stands for "change directory". So you
   would do something like `cd ~/path/to/cloned/repository`
3. The repository contains
   [environment.yml](https://github.com/floriscalkoen/coastpy/environment.yml),
   which is a file that describes the software dependencies. Now create the software environment by running the following command in the terminal/Miniforge prompt:

   ```bash
      mamba env create -f environment.yml
   ```

</details>


## 4. Running the notebooks

Having successfully installed all necessary content and software on your computer, you're
ready to move forward. The [following section](launch.md) will guide you through
running thenotebooks!
