# REBOOT

## Why REBOOT?

- Our old repository became somewhat chaotic and did not follow the KISS
  principle: **Keep It Simple, Stupid**.
- To follow KISS, we no longer use the dashboard.
- We make every lab's code as easy and simple as possible.

Every file or lab—except IMU, SLAM, and machine-learning programs—is
**67 lines or fewer**.

If a file has `GF` in its name, it means that the program is based on a
gap follower.

---

## Documentation

The `documentation` folder contains our daily logs and journals beginning
with Week 2, Day 1.

It contains two versions:

- **OV:** Read this for the most important information and a short overview
  of the day.
- **Full:** Read this when you want complete and detailed information about
  the day.

---

## Running the Code

### 1 | Download the Repository

1. Click the green **Code** button near the top of the repository.
2. Under the **Local** section, click **Open with GitHub Desktop**.
3. Download and install
   [GitHub Desktop](https://desktop.github.com/) if you need.
4. In GitHub Desktop, clone the repository onto your computer.
5. From the menu, select **Repository → Open in Visual Studio Code**.

This opens the cloned GitHub repository in Visual Studio Code.

---

### 2 | Navigate Through the Repository

Open a terminal in Visual Studio Code and use `cd` to move to the folder
containing the program:

```bash
cd folder_name
```

To move back by one folder, use:

```bash
cd ..
```

---

### 3 | Run in the RACECAR Simulator

Once the terminal is in the correct folder, run:

```bash
racecar sim ____.py
```

Replace `____.py` with the name of the program you want to run.

---

## Running the Code on the Physical RACECAR

### 1 | Connect to the RACECAR

Connect your computer to the RACECAR's Wi-Fi network:

```text
SSID: racecar-neo-(__)
Password: ask your instructor
```

Replace `(__)` with your team's number.

---

### 2 | Open the RACECAR Configuration

In the terminal, run:

```bash
racecar cd && cd ../scripts
vim .config
```

Inside Vim:

1. Press `i` to enter insert mode.
2. Change `TEAM_NAME` to your team name.
3. Change the IP address to `10.42.0.1`.
4. Press `Esc`.
5. Type `:wq`, then press `Enter` to save and exit.

Load the updated configuration:

```bash
source .config
```

---

### 3 | Set Up the Connection

Run:

```bash
racecar setup
```

When asked for the password, enter the car's password (ask your instructor).

### 4 | Open Jupyter Notebook

Open a browser and go to:

```text
http://10.42.0.1:8888
```

Jupyter Notebook should open. Locate the folder containing the program.

In the Jupyter terminal, navigate to that folder and run:

```bash
python3 ____.py
```

Replace `____.py` with the name of the program you want to run.

