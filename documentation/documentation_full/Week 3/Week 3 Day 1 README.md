# Quest Log Day 6 — Detailed Documentation

---

## Side Quest and Equipment

During the weekend, three members of our team completed a cooking side quest. We made egg fried rice while completing the “blind, deaf, mute” communication challenge. Completing the activity awarded us three CC coins. We planned to use one coin to buy an encoder from the flash sale because we would need it for the IMU and SLAM labs. We also planned to use one coin to pay back a loan from another team.

---

## Testing Plan

### Test 1:

- Collect data for SLAM
- Create the SLAM map

### Test 2:

- Connect properly to Diana
- Collect additional data
- Test whether the car can sense and follow the line

### Test 3:

- Test the PD line follower
- Observe its steering corrections and stability

### Test 4:

- Adjust and retest the line follower
- If the line follower is not close to completion, test the wall follower
- Try to complete the wall-follower course in under 20 seconds

---

## Testing Goals

Our main goals for the day were to collect data for SLAM, make a SLAM map, connect to the replacement car, and continue testing the line follower. We also wanted to test and adjust the PD line follower. If the line follower was not close to completion, our fallback plan was to test the wall follower and try to complete it in under 20 seconds.

---

## Testing Problems and Results

Our testing time was frustrating and unproductive because a different problem occurred during nearly every 15-minute session.

During the first test, the controller had no battery, so we could not complete the planned work. We then found that our car, Zeus, had a faulty PCB. This problem cost us another testing session.

We received a replacement car named Diana, which the TAs described as one of the best available cars. However, changing cars meant that we would have to adjust our values because the behavior and hardware characteristics of the new car could differ from Zeus.

During the next test, the controller lost power again. Our code also did not work because it had not been uploaded properly to the car. We did not attend the final planned test because of a class-wide administrative situation.

Because of these repeated hardware, battery, setup, and code-upload problems, we did not collect reliable performance data for SLAM, the line follower, or the wall follower. We also could not make a meaningful comparison between controller values on Zeus and Diana.

---

## Preparation and Workflow Limitations

New class rules required phones to be placed in a caddy during preparation time. This affected our existing workflow because some team members used their laptops to connect to the car and run code while using their phones for research and data recording. We will need to reorganize this process by moving research, notes, and data recording to available laptops or shared documents.

The 15-minute testing sessions also made debugging difficult. When a battery, PCB, connection, or upload problem occurred, most or all of the session was used before the car could be tested. This left limited time to observe the car, change values, and test again.

---

## Engineering Exam Preparation

After class, I studied with another student for approximately four hours to prepare for the engineering exam. We reviewed the first few chapters from the course and taught each other concepts that we had forgotten. This preparation was also important because the top five scores would receive one CC coin each, which could help our team obtain equipment for the Quest Lab.
