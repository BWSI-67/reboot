# Quest Log Day 4 — Wall Follower Testing

The main goal for Day 4 was to test three different wall-following controllers: **bang-bang control, PD control, and single-setpoint proportional control**. We wanted to compare how reliably each controller could complete the course, how quickly it could finish, and how many times the RACECAR hit the wall. After testing all three, we planned to choose the best-working controller and continue improving it.

---

## Test Trial 1

For the first trial, we expected to test the bang-bang, PD, and single-setpoint proportional wall followers. Bang-bang control was used as the simplest controller and as a baseline for comparing the other methods.

The bang-bang wall follower completed the course in **44 seconds with one hit**. The PD controller did not finish and had a few hits. The single-setpoint proportional controller also did not finish and had a few hits.

Based on these results, bang-bang was the only controller that successfully completed the course. However, we decided to continue working on the PD and single-setpoint proportional controllers because they had more potential for smoother and faster wall following. We kept the completed bang-bang controller as our baseline.

---

## Test Trial 2

For the second trial, we expected both the PD and proportional controllers to work better after adjustment. We tuned the PD values and tested increasing and decreasing the setpoint used by the proportional controller.

The single-setpoint proportional controller completed the course in **29 seconds with zero hits**. This was a major improvement compared with the first trial and was also faster and cleaner than the bang-bang baseline. The PD controller still did not finish and had a few hits.

After this trial, we decided to continue working on the single-setpoint proportional controller. The PD controller was kept as a backup, while bang-bang remained useful as the original completed baseline.

---

## Test Trial 3

For the third trial, we planned to test the single-setpoint proportional wall follower again after adjusting some of its values. If the wall-following test was successful, our next goal was to begin testing the line follower.

After the adjustments, the single-setpoint proportional controller completed the wall follower trial and received **2 cc coins**. This confirmed that it was the best-working wall follower from the three approaches we tested during the day.

---

## Overall Result

The testing process showed a clear improvement from the original bang-bang baseline to the single-setpoint proportional controller. Bang-bang completed the course in **44 seconds with one hit**, while the improved proportional controller completed it in **29 seconds with no hits** during Trial 2 and completed the final wall follower trial during Trial 3. The PD controller did not finish its trials, so it remained a backup rather than the final selected controller. After completing the wall follower trial, the next planned task was line-follower testing.
